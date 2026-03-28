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
    "sdr_linked": False
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
        await asyncio.sleep(0.02) # 50 FPS broadcast

class SpectreAegisController:
    def __init__(self, use_digital_twin=False):
        logging.info("Initializing Spectre AEGIS Controller...")
        self.sdr = SDRHandler(use_digital_twin=use_digital_twin, center_freq=BANDS[0])
        self.dsp = FastSpectrogramProcessor()
        self.cnn = InferenceEngine()
        self.rl_agent = RLController(action_dim=len(BANDS))
        self.current_channel_idx = 0
        self.run_loop = True
        
    def execute_cycle(self):
        t_start = time.perf_counter()
        
        # Sense
        iq_data = self.sdr.capture_iq()
        
        # Format Waterfall Slice for UI (downsample for bandwidth)
        waterfall_slice = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(iq_data[:1024]))) + 1e-12)
        global_telemetry["waterfall_slice"] = waterfall_slice.tolist()
        
        # DSP
        spectrogram = self.dsp.process_iq(iq_data)
        
        # AI CNN (Multiclass)
        is_busy, class_name, confidence = self.cnn.predict(spectrogram)
        
        global_telemetry["is_busy"] = is_busy
        global_telemetry["class_name"] = class_name
        global_telemetry["confidence"] = round(confidence, 3)
        
        # History
        self.rl_agent.push_state(is_busy)
        
        # Extract Q-Values for analytical dash
        with torch.no_grad() if 'torch' in globals() else np.errstate():
            seq = self.rl_agent.get_current_sequence()
            q_vals, _ = self.rl_agent.q_network(seq)
            global_telemetry["q_values"] = q_vals[0].cpu().numpy().tolist()

        if is_busy:
            action = self.rl_agent.select_action()
            new_freq = BANDS[action]
            self.sdr.set_frequency(new_freq)
            self.current_channel_idx = action
            reward = self.rl_agent.compute_reward(is_busy=True, action_channel=action)
        else:
            tx_data = np.zeros(1024, dtype=complex)
            self.sdr.transmit(tx_data)
            reward = self.rl_agent.compute_reward(is_busy=False, action_channel=self.current_channel_idx)
            
        t_end = time.perf_counter()
        lat = (t_end - t_start) * 1000
        
        global_telemetry["latency_ms"] = round(lat, 2)
        global_telemetry["channel_idx"] = self.current_channel_idx
        global_telemetry["reward"] = reward
        return lat, reward

    def controller_thread(self):
        while self.run_loop:
            self.execute_cycle()
            time.sleep(0.01)

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
