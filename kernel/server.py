import asyncio
import websockets
from .audio_processor import AudioProcessor

async def handle_client(websocket):
    processor = AudioProcessor()
    print("Client connected.")
    
    try:
        async for message in websocket:
            detected_label = processor.process_chunk(message)
            if detected_label:
                await websocket.send(f"DETECTED:{detected_label}")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

async def main():
    print(f"Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
