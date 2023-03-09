import sys
import oneflow as flow
#flow.mock_torch.enable()

import torch
sys.path.append("../..")
import whisper # torch
import time

if __name__ == "__main__":
    print("is PyTorch or not:", not flow.mock_torch.is_enabled())

    dims_tiny = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4}
    dims = whisper.ModelDimensions(**dims_tiny)
    model = whisper.Whisper(dims)

    checkpoint = torch.load("./tiny.pt")
    params = checkpoint["model_state_dict"] 
    model.load_state_dict(params)
    model = model.cuda()

    start = time.time()
    # import pdb;pdb.set_trace()
    for i in range(0, 5):
        result = model.transcribe("micro-machines.wav")
    end = time.time()
    print(f"{end-start} seconds", result["text"])
