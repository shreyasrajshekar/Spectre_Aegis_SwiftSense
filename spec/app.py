import time
import logging
import argparse
import numpy as np
import threading
import asyncio
import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

from core.sdr_handler import SDRHandler
from core.dsp import FastSpectrogramProcessor
from ai.sensing_cnn import InferenceEngine
from ai.decision_d3qn import RLController
from core.db_logger import build_logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

BANDS = [2.4e9 + (i * 10e6) for i in range(10)]

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
    "pls_score": 100,
    "current_layer": "Legacy",
    "layer_labels": []
}

LAYER_CONFIGS = {
    "Legacy": {"start": 0.433, "step": 0.6, "unit": "GHz"},   # 433 MHz → ~5.9 GHz ISM
    "cmWave": {"start": 7.0, "step": 1.7, "unit": "GHz"},      # 7 GHz → 24 GHz
    "Sub-THz": {"start": 90.0, "step": 21.0, "unit": "GHz"},   # 90 GHz → 300 GHz
    "THz": {"start": 300.0, "step": 970.0, "unit": "GHz"}       # 300 GHz → 10 THz
}
connected_clients = []

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
                layer = cmd_data.get('layer', 'cmWave')
                if layer in LAYER_CONFIGS:
                    global_telemetry["current_layer"] = layer
                    if aegis.sdr.simulator:
                        aegis.sdr.simulator.current_layer = layer
                    logging.info(f"6G Air Interface: Switched to {layer} band.")
            elif cmd_data.get('cmd') == 'toggle_mode':
                aegis.sdr.use_digital_twin = cmd_data.get('val', False)
                if aegis.sdr.use_digital_twin:
                    if not aegis.sdr.simulator:
                        from core.digital_twin import DigitalTwinSimulator
                        aegis.sdr.simulator = DigitalTwinSimulator(aegis.sdr.buffer_size)
            elif cmd_data.get('cmd') == 'toggle_optimizer':
                is_max_throughput = cmd_data.get('val', False)
                if is_max_throughput:
                    aegis.rl_agent.epsilon = 0.5 
            elif cmd_data.get('cmd') == 'set_channel':
                ch_idx = cmd_data.get('idx', 0)
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
            elif cmd_data.get('cmd') == 'toggle_manual_mode':
                aegis.manual_override = cmd_data.get('val', False)
                global_telemetry["manual_mode"] = aegis.manual_override
                logging.info(f"Control Mode Switched: {'MANUAL' if aegis.manual_override else 'AUTO'}")
            elif cmd_data.get('cmd') == 'toggle_pause':
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
        
    def execute_cycle(self):
        t_start = time.perf_counter()
        
        # Sense
        iq_data = self.sdr.capture_iq()
        if iq_data is None:
            logging.warning("SDR capture returned None. Skipping cycle.")
            return
        
        # Format Waterfall Slice for UI (downsample for bandwidth)
        psd_data = np.abs(np.fft.fftshift(np.fft.fft(iq_data[:1024]))) + 1e-12
        waterfall_slice = 10 * np.log10(psd_data)
        if hasattr(waterfall_slice, "tolist"):
            global_telemetry["waterfall_slice"] = waterfall_slice.tolist()
        else:
            global_telemetry["waterfall_slice"] = [float(x) for x in waterfall_slice]
        
        # DSP
        spectrogram = self.dsp.process_iq(iq_data)
        
        # AI CNN (Semantic sensing)
        is_busy_ai, class_name, confidence, priority = self.cnn.predict(spectrogram)
        
        # Power Squelch Check
        avg_pwr = 10 * np.log10(np.mean(np.abs(iq_data)**2) + 1e-12)
        is_busy_pwr = avg_pwr > self.power_threshold
        
        is_busy = bool(is_busy_ai or is_busy_pwr)
        
        # Update Telemetry & History
        global_telemetry["is_busy"] = is_busy
        global_telemetry["manual_mode"] = self.manual_override
        global_telemetry["class_name"] = str(class_name) if is_busy else "Idle"
        global_telemetry["confidence"] = round(float(confidence), 3)
        global_telemetry["priority"] = priority
        global_telemetry["active_beam"] = self.current_beam_idx

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
            
        # 6G ISAC: Fetch spatial radar map
        global_telemetry["radar_map"] = self.sdr.get_sensing_radar()
        
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
        global_telemetry["channel_idx"] = int(self.current_channel_idx)
        
        # Update dynamic frequency labels for HUD
        layer = global_telemetry["current_layer"]
        conf = LAYER_CONFIGS[layer]
        global_telemetry["layer_labels"] = [f"{(conf['start'] + i*conf['step']):.1f}{conf['unit']}" for i in range(10)]
        
        # Publish to Bridge (ORAN RIC)
        self.ric_bridge.publish_metrics(global_telemetry)

        # Persist telemetry row to MySQL (non-blocking — queued async)
        self.db.log(global_telemetry)

        return lat, reward

    def controller_thread(self):
        while self.run_loop:
            if not self.is_paused:
                self.execute_cycle()
                
            # 6G Green Networking: Adaptive Sleep (Eco-Mode)
            # If prediction horizon is low (< 0.2 occupancy probability)
            # we can sleep longer to save energy
            pred = float(global_telemetry.get("prediction_horizon", 0.0))
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

# Ensure static folder exists
os.makedirs("static", exist_ok=True)
fastapi_app.mount("/", StaticFiles(directory="static", html=True), name="static")

@fastapi_app.on_event("startup")
async def startup_event():
    # Start telemetry broadcaster
    asyncio.create_task(broadcast_telemetry())
    
if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--twin", action="store_true",
                        help="Launch with Digital Twin SDR simulator")
    # MySQL logging (all optional – omit to disable DB logging)
    parser.add_argument("--db-host", default=None, help="MySQL host (e.g. localhost)")
    parser.add_argument("--db-port", type=int, default=3306, help="MySQL port (default 3306)")
    parser.add_argument("--db-user", default=None, help="MySQL username")
    parser.add_argument("--db-pass", default=None, help="MySQL password")
    parser.add_argument("--db-name", default=None, help="MySQL database / schema name")
    args = parser.parse_args()

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
