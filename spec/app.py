import time
import logging
import argparse
import numpy as np
import threading
import asyncio
import json
import uvicorn
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import os

from core.sdr_handler import SDRHandler
from core.dsp import FastSpectrogramProcessor
from ai.sensing_cnn import InferenceEngine
from ai.decision_d3qn import RLController
from core.db_logger import build_logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Legacy ISM channels within PlutoSDR hardware range (325 MHz – 3.8 GHz)
# 433 MHz, 868 MHz, 915 MHz, 1.2 GHz, 1.5 GHz, 1.8 GHz, 2.1 GHz, 2.4 GHz, 3.0 GHz, 3.5 GHz
BANDS = [433e6, 868e6, 915e6, 1.2e9, 1.5e9, 1.8e9, 2.1e9, 2.4e9, 3.0e9, 3.5e9]

fastapi_app = FastAPI()

# Global state for telemetry
global_telemetry = {
    "latency_ms": 0.0,
    "class_name": "Idle",
    "confidence": 0.0,
    "is_busy": False,
    "channel_idx": 0,
    "reward": 0.0,
    "q_values": [],
    "waterfall_slice": [],
    "sdr_linked": False,
    "reasoning_msg": "Initializing Aegis...",
    "event_trigger": None,
    "conf_threshold": 0.5,
    "pwr_threshold": -60.0,
    "manual_mode": False,
    "is_paused": False,
    "active_beam": 0,
    "priority": 0.0,
    "prediction_horizon": 0.0,
    "prediction_confidence": 0.0,
    "trend": "stable",
    "forecast_array": [0.0, 0.0, 0.0],
    "radar_map": [],
    "eco_saving": 0,
    "last_hop_db": "0.00",
    "pls_score": 100,
    "current_layer": "Legacy",
    "layer_labels": ["433M", "868M", "915M", "1.2G", "1.5G", "1.8G", "2.1G", "2.4G", "3.0G", "3.5G"],
    "d3qn_loss": 0.0
}

# Legacy ISM band only — all frequencies within ADALM-Pluto hardware range
LAYER_CONFIG = {"start": 0.433, "step": 0.3, "unit": "GHz"}
connected_clients = []
aegis = None

@fastapi_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            cmd_data = json.loads(data)
            
            if cmd_data.get('cmd') == 'exit':
                logging.warning("User requested remote shutdown. Terminating...")
                os._exit(0)
            elif cmd_data.get('cmd') == 'switch_layer':
                pass  # Only Legacy band supported — layer switching disabled
            elif cmd_data.get('cmd') == 'toggle_mode':
                if aegis:
                    aegis.sdr.use_digital_twin = cmd_data.get('val', False)
                    if aegis.sdr.use_digital_twin:
                        if not aegis.sdr.simulator:
                            from core.digital_twin import DigitalTwinSimulator
                            aegis.sdr.simulator = DigitalTwinSimulator(aegis.sdr.buffer_size)
            elif cmd_data.get('cmd') == 'toggle_optimizer':
                is_max_throughput = cmd_data.get('val', False)
                if is_max_throughput and aegis:
                    aegis.rl_agent.epsilon = 0.5 
            elif cmd_data.get('cmd') == 'set_channel':
                ch_idx = cmd_data.get('idx', 0)
                if aegis:
                    aegis.manual_override = True
                    aegis.current_channel_idx = ch_idx
                    aegis.sdr.set_frequency(BANDS[ch_idx])
                    logging.info(f"Manual Override: Forced channel {ch_idx}")
            elif cmd_data.get('cmd') == 'update_params':
                conf = cmd_data.get('conf')
                pwr = cmd_data.get('pwr')
                ai_thresh = cmd_data.get('ai_thresh')
                hist_w = cmd_data.get('hist_w')
                uncert_w = cmd_data.get('uncert_w')
                if aegis:
                    if conf is not None:
                        aegis.cnn.confidence_threshold = conf
                        global_telemetry["conf_threshold"] = conf
                    if pwr is not None:
                        aegis.power_threshold = pwr
                        global_telemetry["pwr_threshold"] = pwr
                    if ai_thresh is not None:
                        aegis.rl_agent.collision_threshold = ai_thresh
                    if hist_w is not None:
                        aegis.rl_agent.history_weight = hist_w
                    if uncert_w is not None:
                        aegis.rl_agent.uncertainty_weight = uncert_w
            elif cmd_data.get('cmd') == 'set_network_slice':
                slice_val = cmd_data.get('val', 'none')
                global_telemetry['network_slice'] = slice_val
                if aegis:
                    if slice_val == 'urllc':
                        aegis.rl_agent.collision_threshold = 0.90
                        aegis.rl_agent.history_weight = 0.90
                        aegis.rl_agent.epsilon = 0.1
                        logging.info("Network Slice: URLLC [Lat/Rel Prioritized]")
                    elif slice_val == 'embb':
                        aegis.rl_agent.collision_threshold = 0.50
                        aegis.rl_agent.history_weight = 0.50
                        aegis.rl_agent.epsilon = 0.4
                        logging.info("Network Slice: eMBB [Throughput Prioritized]")
                    elif slice_val == 'mmtc':
                        aegis.rl_agent.collision_threshold = 0.70
                        aegis.rl_agent.history_weight = 0.80
                        logging.info("Network Slice: mMTC [Energy Efficiency Prioritized]")
                    else:
                        logging.info("Network Slice: Default Context")
            elif cmd_data.get('cmd') == 'toggle_manual_mode':
                if aegis:
                    aegis.manual_override = cmd_data.get('val', False)
                    global_telemetry["manual_mode"] = aegis.manual_override
                    logging.info(f"Control Mode Switched: {'MANUAL' if aegis.manual_override else 'AUTO'}")
            elif cmd_data.get('cmd') == 'toggle_pause':
                if aegis:
                    aegis.is_paused = cmd_data.get('val', False)
                    global_telemetry["is_paused"] = aegis.is_paused
                    logging.info(f"Scan Loop {'PAUSED' if aegis.is_paused else 'RESUMED'}")
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
    except Exception as e:
        logging.error(f"WS error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)

async def broadcast_telemetry():
    while True:
        try:
            global_telemetry["sdr_linked"] = True if aegis.sdr.use_digital_twin else (aegis.sdr.sdr is not None)
        except NameError:
            pass

        if connected_clients:
            msg = json.dumps(global_telemetry)
            dead_clients = []
            for client in connected_clients:
                try:
                    await client.send_text(msg)
                except Exception:
                    dead_clients.append(client)
            
            for dc in dead_clients:
                if dc in connected_clients:
                    connected_clients.remove(dc)
        await asyncio.sleep(0.1) 

class RICBridge:
    """Simulates a 6G O-RAN E2 Interface over ZeroMQ."""
    def __init__(self, port=5555):
        import zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        try:
            self.socket.bind(f"tcp://*:{port}")
            logging.info(f"6G O-RAN RIC xApp Bridge active on ZMQ port {port}")
        except Exception as e:
            logging.warning(f"O-RAN Bridge: Port {port} in use. RIC Telemetry disabled. ({e})")

    def publish_metrics(self, data):
        """Sends E2-SM-LIKE metrics to the Near-RT RIC."""
        try:
            self.socket.send_json(data, zmq.NOBLOCK)
        except Exception:
            pass

class SpectreAegisController:
    def __init__(self, use_digital_twin=False, db_host=None, db_user=None,
                 db_pass=None, db_name=None, db_port=3306):
        logging.info("Initializing Spectre AEGIS Controller...")
        self.sdr = SDRHandler(use_digital_twin=use_digital_twin, center_freq=BANDS[0])
        if self.sdr.use_digital_twin:
            logging.info("Spectre AEGIS operational in MOCK MODE (Simulated Hardware)")
        else:
            logging.info("Spectre AEGIS operational in HARDWARE MODE (PlutoSDR)")
        self.dsp = FastSpectrogramProcessor()
        self.cnn = InferenceEngine()
        self.rl_agent = RLController(action_dim=40) # 10 Freqs * 4 Beams
        self.current_channel_idx = 0
        self.current_beam_idx = 0
        self.ric_bridge = RICBridge()
        self.run_loop = True
        self.manual_override = False
        self.power_threshold = -60.0
        self.is_paused = False
        # MySQL telemetry sink (NullLogger if no credentials supplied)
        self.db = build_logger(
            host=db_host, user=db_user,
            password=db_pass, database=db_name, port=db_port
        )
        self.last_freq = BANDS[0]
        self.total_energy_score = 50.0 # Start at 50% efficiency baseline
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def execute_cycle(self):
        t_start = time.perf_counter()
        
        # Sense
        iq_data_raw = self.sdr.capture_iq()
        t_cap = time.perf_counter()
        if iq_data_raw is None:
            logging.warning("SDR capture returned None. Skipping cycle.")
            return

        # Ensure single channel numpy array for legacy processing
        if isinstance(iq_data_raw, (list, tuple)) and len(iq_data_raw) >= 2:
            iq_data_np = np.asarray(iq_data_raw[0])
        else:
            iq_data_np = np.asarray(iq_data_raw)

        # Move to GPU once to avoid redundant transfers
        iq_data_tensor = torch.from_numpy(iq_data_np).to(device=self.device, dtype=torch.complex64)
        t_tensor = time.perf_counter()

        # Format Waterfall Slice for UI (downsample for bandwidth)
        psd_data = np.abs(np.fft.fftshift(np.fft.fft(iq_data_np[:1024]))) + 1e-12
        waterfall_slice = 10 * np.log10(psd_data)
        if hasattr(waterfall_slice, "tolist"):
            global_telemetry["waterfall_slice"] = waterfall_slice.tolist()
        else:
            global_telemetry["waterfall_slice"] = [float(x) for x in waterfall_slice]
        t_psd = time.perf_counter()
        
        # DSP (uses Tensor directly)
        spectrogram = self.dsp.process_iq(iq_data_tensor)
        t_dsp = time.perf_counter()
        
        # AI CNN (Semantic sensing)
        is_busy_ai, class_name, confidence, priority = self.cnn.predict(spectrogram)
        t_cnn = time.perf_counter()
        
        # Power Squelch Check
        avg_pwr = 10 * np.log10(np.mean(np.abs(iq_data_np)**2) + 1e-12)
        is_busy_pwr = avg_pwr > self.power_threshold
        
        is_busy = bool(is_busy_ai or is_busy_pwr)
        
        # History: Push (Occupancy, Power, Priority)
        self.rl_agent.push_state(is_busy, avg_pwr, priority)
        
        # 6G Temporal Twin: Lead-time Prediction
        pred_conf, trend, forecast = self.rl_agent.predict_future_occupancy()
        prediction = forecast[0]  # 100ms prediction is the immediate action trigger
        
        global_telemetry["prediction_horizon"] = round(prediction, 3)
        global_telemetry["prediction_confidence"] = round(pred_conf, 3)
        global_telemetry["trend"] = trend
        global_telemetry["forecast_array"] = [round(f, 3) for f in forecast]
        
        # Proactive Vacation Logic: If predicted occupancy > dynamic collision_threshold, trigger early hop
        is_collision_predicted = prediction > self.rl_agent.collision_threshold
        
        global_telemetry["event_trigger"] = "predictive_vacate" if is_collision_predicted else None
        
        # Extract Q-Values for analytical dash
        with torch.no_grad() if 'torch' in globals() else np.errstate():
            seq = self.rl_agent.get_current_sequence()
            q_vals, _ = self.rl_agent.q_network(seq)
            global_telemetry["q_values"] = q_vals[0].cpu().numpy().tolist()

        if is_busy or is_collision_predicted:
            if is_busy: global_telemetry["event_trigger"] = "collision"
            
            if not self.manual_override:
                action_ch, action_beam = self.rl_agent.select_action()
                
                # Act
                self.sdr.set_frequency(BANDS[action_ch])
                self.sdr.set_beam_direction(action_beam)
                
                self.current_channel_idx = action_ch
                self.current_beam_idx = action_beam
                
                reason = "VACATE" if is_collision_predicted else "REACTIVE"
                global_telemetry["reasoning_msg"] = f"{reason}: PU '{class_name}' @ CH{self.current_channel_idx} -> Hop CH{action_ch} BEAM{action_beam}"
            else:
                global_telemetry["reasoning_msg"] = f"Manual Override active. PU detected but hold frequency."
            reward = self.rl_agent.compute_reward(is_busy=is_busy, action_channel=self.current_channel_idx)
        else:
            if not self.manual_override:
                global_telemetry["reasoning_msg"] = f"Sensing... CH{self.current_channel_idx} IDLE"
            else:
                global_telemetry["reasoning_msg"] = f"Manual Lock on CH{self.current_channel_idx}"
            
            tx_data = np.zeros(1024, dtype=complex)
            self.sdr.transmit(tx_data)
            reward = self.rl_agent.compute_reward(is_busy=False, action_channel=self.current_channel_idx)
            
        # ── Final Telemetry Refresh (Reflect Action Taken) ───────────────────
        global_telemetry["is_busy"] = is_busy
        global_telemetry["manual_mode"] = self.manual_override
        global_telemetry["class_name"] = str(class_name) if is_busy else "Idle"
        global_telemetry["confidence"] = round(float(confidence), 3)
        global_telemetry["priority"] = priority
        global_telemetry["active_beam"] = self.current_beam_idx
        global_telemetry["channel_idx"] = int(self.current_channel_idx)
        # ────────────────────────────────────────────────────────────────────
            
        # ── Reinforcement Learning Training ───────────────────────────────────
        self.rl_agent.push_transition(reward)
        loss = self.rl_agent.update()
        global_telemetry["d3qn_loss"] = round(float(loss), 4)

        # ── Energy Saving Calculation (Frequency Transition Delta) ──────────
        current_f = BANDS[self.current_channel_idx]
        if current_f != self.last_freq:
            import math
            # S_dB = 20 * log10(f_prev / f_curr)
            # Hopping to lower freq = positive saving
            try:
                delta_db = 20 * math.log10(self.last_freq / current_f)
                global_telemetry["last_hop_db"] = f"{delta_db:+.2f}"
                
                # Update cumulative eco_saving score (normalized)
                # Max hop delta is ~22.5 dB (5.8G to 433M). Map -25..+25 to -5..+5 weight.
                weight = max(-5, min(5, delta_db / 5.0)) 
                self.total_energy_score = max(0, min(100, self.total_energy_score + weight))
                global_telemetry["eco_saving"] = int(self.total_energy_score)
                
                if delta_db > 0:
                    logging.info(f"Energy Saving: {delta_db:+.2f} dB transition (Propagation bonus)")
                else:
                    logging.info(f"Energy Cost: {delta_db:+.2f} dB transition (Higher frequency propagation loss)")
            except (ValueError, ZeroDivisionError):
                pass
            
            self.last_freq = current_f
        # ────────────────────────────────────────────────────────────────────
            
        # 6G ISAC: Fetch spatial radar map (Reuse GPU Tensor to save latency)
        global_telemetry["radar_map"] = self.sdr.get_sensing_radar(iq_data_tensor)
        t_radar = time.perf_counter()
        
        # 6G PLS: Simulate Physical Layer Security Trust Score
        # Rogue signals (low probability or unknown classes) reduce trust
        current_pls = float(global_telemetry.get("pls_score", 100))
        if is_busy and confidence < 0.6:
            global_telemetry["pls_score"] = int(max(0, current_pls - 5))
        else:
            global_telemetry["pls_score"] = int(min(100, current_pls + 1))

        t_end = time.perf_counter()
        lat = (t_end - t_start) * 1000
        
        global_telemetry["reward"] = round(float(reward), 3)
        global_telemetry["latency_ms"] = float(round(lat, 2))
        
        # Update dynamic frequency labels for HUD based on BANDS
        global_telemetry["layer_labels"] = [f"{b/1e9:.1f}G" if b >= 1e9 else f"{b/1e6:.0f}M" for b in BANDS]
        
        # Publish to Bridge (ORAN RIC)
        self.ric_bridge.publish_metrics(global_telemetry)

        # Persist telemetry row to MySQL (non-blocking — queued async)
        self.db.log(global_telemetry)

        t_end_total = time.perf_counter()
        
        logging.info(f"[PROFILER] rx:{int((t_cap-t_start)*1000)}ms ten:{int((t_tensor-t_cap)*1000)}ms psd:{int((t_psd-t_tensor)*1000)}ms dsp:{int((t_dsp-t_psd)*1000)}ms cnn:{int((t_cnn-t_dsp)*1000)}ms rad:{int((t_radar-t_cnn)*1000)}ms total:{int((t_end_total-t_start)*1000)}ms")

        return lat, reward

    def controller_thread(self):
        while self.run_loop:
            if not self.is_paused:
                try:
                    self.execute_cycle()
                except Exception as e:
                    logging.error(f"Error in controller cycle: {e}")
                    time.sleep(1.0) # Recovery wait period before next attempt
                
            # 6G Green Networking: Adaptive Sleep (Eco-Mode) + Slice Tying
            pred = float(global_telemetry.get("prediction_horizon", 0.0))
            slice_mode = global_telemetry.get("network_slice", "embb")
            
            if slice_mode == 'urllc':
                sleep_time = 0.01 # 100Hz Sensing (Max Performance)
                global_telemetry["eco_saving"] = 0
            elif slice_mode == 'mmtc':
                sleep_time = 0.4  # Extreme Low Power IoT Mode
                global_telemetry["eco_saving"] = 90
            else:
                # eMBB or Default: Adaptive Sleep
                if pred < 0.15:
                    sleep_time = 0.2 # 5Hz sensing (Eco MAX)
                    global_telemetry["eco_saving"] = 75
                elif pred < 0.35:
                    sleep_time = 0.1 # 10Hz sensing (Eco Light)
                    global_telemetry["eco_saving"] = 50
                else:
                    sleep_time = 0.05 # 20Hz sensing (Active)
                    global_telemetry["eco_saving"] = 0
                
            time.sleep(sleep_time)

# Ensure static folder exists and has at least an empty style.css if missing
_static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(_static_dir, exist_ok=True)

# Explicitly serve the dashboard at /
@fastapi_app.get("/")
async def get_dashboard():
    with open(os.path.join(_static_dir, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Mount remaining static assets (CSS, JS)
fastapi_app.mount("/", StaticFiles(directory=_static_dir), name="static")

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    # Start telemetry broadcaster
    task = asyncio.create_task(broadcast_telemetry())
    yield
    task.cancel()

fastapi_app.router.lifespan_context = app_lifespan

if __name__ == "__main__":

    # Load .env file if present (DB credentials live there, not in source)
    _env_path = os.path.join(os.path.dirname(__file__), '.env')
    _env = {}
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _v = _line.split('=', 1)
                    _env[_k.strip()] = _v.strip()

    parser = argparse.ArgumentParser()
    parser.add_argument("--twin", action="store_true",
                        help="Launch with Digital Twin SDR simulator")
    # MySQL logging — reads from .env by default, CLI args override.
    parser.add_argument("--db-host", default=_env.get("DB_HOST", "localhost"),  help="MySQL host")
    parser.add_argument("--db-port", type=int, default=int(_env.get("DB_PORT", 3306)), help="MySQL port")
    parser.add_argument("--db-user", default=_env.get("DB_USER", "root"),       help="MySQL username")
    parser.add_argument("--db-pass", default=_env.get("DB_PASS", ""),           help="MySQL password")
    parser.add_argument("--db-name", default=_env.get("DB_NAME", "aegis"),      help="MySQL database name")
    parser.add_argument("--no-db",   action="store_true",  help="Disable DB logging entirely")
    args = parser.parse_args()

    # --no-db flag → wipe credentials so build_logger returns NullLogger
    if args.no_db:
        args.db_host = args.db_user = args.db_pass = args.db_name = None
    else:
        logging.info(f"DB Logging → {args.db_user}@{args.db_host}:{args.db_port}/{args.db_name}")

    aegis = SpectreAegisController(
        use_digital_twin=args.twin,
        db_host=args.db_host,
        db_user=args.db_user,
        db_pass=args.db_pass,
        db_name=args.db_name,
        db_port=args.db_port,
    )
    # Start the hardware RT loop in a separate thread so Uvicorn runs unblocked on main
    t = threading.Thread(target=aegis.controller_thread, daemon=True)
    t.start()

    logging.info("Starting Aegis Dashboard Server at http://localhost:8000")
    try:
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="error")
    finally:
        aegis.run_loop = False
        aegis.db.shutdown()

