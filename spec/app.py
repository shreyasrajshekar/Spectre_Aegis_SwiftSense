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
    "is_paused": False
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
                import os, signal
                os.kill(os.getpid(), signal.SIGTERM)
                
            elif cmd_data.get('cmd') == 'toggle_twin':
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
                if conf is not None:
                    aegis.cnn.confidence_threshold = conf
                    global_telemetry["conf_threshold"] = conf
                if pwr is not None:
                    aegis.power_threshold = pwr
                    global_telemetry["pwr_threshold"] = pwr
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
        await asyncio.sleep(0.1) # 10 FPS broadcast is plenty for HUD

class SpectreAegisController:
    def __init__(self, use_digital_twin=False):
        logging.info("Initializing Spectre AEGIS Controller...")
        self.sdr = SDRHandler(use_digital_twin=use_digital_twin, center_freq=BANDS[0])
        self.dsp = FastSpectrogramProcessor()
        self.cnn = InferenceEngine()
        self.rl_agent = RLController(action_dim=len(BANDS))
        self.current_channel_idx = 0
        self.run_loop = True
        self.manual_override = False
        self.power_threshold = -60.0
        self.is_paused = False
        
    def execute_cycle(self):
        t_start = time.perf_counter()
        
        # Sense
        iq_data = self.sdr.capture_iq()
        
        # Format Waterfall Slice for UI (downsample for bandwidth)
        psd_data = np.abs(np.fft.fftshift(np.fft.fft(iq_data[:1024]))) + 1e-12
        waterfall_slice = 10 * np.log10(psd_data)
        if hasattr(waterfall_slice, "tolist"):
            global_telemetry["waterfall_slice"] = waterfall_slice.tolist()
        else:
            global_telemetry["waterfall_slice"] = [float(x) for x in waterfall_slice]
        
        # DSP
        spectrogram = self.dsp.process_iq(iq_data)
        
        # AI CNN (Multiclass)
        is_busy_ai, class_name, confidence = self.cnn.predict(spectrogram)
        
        # Power Squelch Check
        avg_pwr = 10 * np.log10(np.mean(np.abs(iq_data)**2) + 1e-12)
        is_busy_pwr = avg_pwr > self.power_threshold
        
        is_busy = bool(is_busy_ai or is_busy_pwr)
        
        global_telemetry["is_busy"] = is_busy
        global_telemetry["manual_mode"] = self.manual_override
        global_telemetry["class_name"] = str(class_name) if is_busy_ai else "Noise"
        global_telemetry["confidence"] = round(float(confidence), 3)
        global_telemetry["event_trigger"] = None

        # History
        self.rl_agent.push_state(is_busy)
        
        # Extract Q-Values for analytical dash
        with torch.no_grad() if 'torch' in globals() else np.errstate():
            seq = self.rl_agent.get_current_sequence()
            q_vals, _ = self.rl_agent.q_network(seq)
            global_telemetry["q_values"] = q_vals[0].cpu().numpy().tolist()

        if is_busy:
            global_telemetry["event_trigger"] = "collision"
            if not self.manual_override:
                action = self.rl_agent.select_action()
                new_freq = BANDS[action]
                self.sdr.set_frequency(new_freq)
                self.current_channel_idx = action
                global_telemetry["reasoning_msg"] = f"PU '{class_name}' @ CH{self.current_channel_idx} -> Hopping to CH{action}"
            else:
                global_telemetry["reasoning_msg"] = f"Manual Override active. PU detected but hold frequency."
            reward = self.rl_agent.compute_reward(is_busy=True, action_channel=self.current_channel_idx)
        else:
            if not self.manual_override:
                global_telemetry["reasoning_msg"] = f"Sensing... CH{self.current_channel_idx} IDLE"
            else:
                global_telemetry["reasoning_msg"] = f"Manual Lock on CH{self.current_channel_idx}"
            
            tx_data = np.zeros(1024, dtype=complex)
            self.sdr.transmit(tx_data)
            reward = self.rl_agent.compute_reward(is_busy=False, action_channel=self.current_channel_idx)
            
        t_end = time.perf_counter()
        lat = (t_end - t_start) * 1000
        
        global_telemetry["reward"] = round(float(reward), 3)
        global_telemetry["latency_ms"] = float(round(lat, 2))
        global_telemetry["channel_idx"] = int(self.current_channel_idx)
        return lat, reward

    def controller_thread(self):
        while self.run_loop:
            if not self.is_paused:
                self.execute_cycle()
            time.sleep(0.05) # 20 Hz loop is sufficient and saves CPU

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
    parser.add_argument("--twin", action="store_true", help="Launch with Digital Twin SDR simulator")
    args = parser.parse_args()
    
    aegis = SpectreAegisController(use_digital_twin=args.twin)
    # Start the hardware RT loop in a separate thread so Uvicorn runs unblocked on main
    t = threading.Thread(target=aegis.controller_thread, daemon=True)
    t.start()
    
    logging.info("Starting Aegis Dashboard Server at http://localhost:8000")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="error")
