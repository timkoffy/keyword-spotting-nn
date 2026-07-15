class AudioStreamerProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const input = inputs[0][0];
        if (input) {
            const int16Array = new Int16Array(input.length);
            for (let i = 0; i < input.length; i++) {
                const s = Math.max(-1, Math.min(1, input[i]));
                int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            this.port.postMessage(int16Array.buffer, [int16Array.buffer]);
        }
        return true;
    }
}
registerProcessor('audio-streamer-processor', AudioStreamerProcessor);
