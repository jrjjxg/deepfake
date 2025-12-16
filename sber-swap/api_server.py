import argparse
import base64
import os
import traceback

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# Local imports (paths are relative to sber-swap root)
from network.AEI_Net import AEI_Net
from arcface_model.iresnet import iresnet100
from insightface_func.face_detect_crop_multi import Face_detect_crop
from coordinate_reg.image_infer import Handler
from utils.inference.image_processing import crop_face, get_final_image
from utils.inference.video_processing import get_target
from utils.inference.core import model_inference


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode BGR image to base64 jpeg string."""
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Failed to encode result image")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


class GhostModelService:
    """Wraps sber-swap (GHOST) models for HTTP serving."""

    def __init__(self, cpu: bool = False, crop_size: int = 224):
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.cpu_flag = cpu
        self.crop_size = crop_size
        self.use_cuda = False
        self.device = None
        self.app = None
        self.G = None
        self.netArc = None
        self.handler = None
        self.is_initialized = False
        self._initialize()

    def _initialize(self):
        try:
            self.use_cuda = torch.cuda.is_available() and (not self.cpu_flag)
            self.device = torch.device('cuda' if self.use_cuda else 'cpu')
            ctx_id = 0 if self.use_cuda else -1
            
            if not self.use_cuda:
                print("CUDA not available or forced CPU, GHOST will run on CPU (slow).")
            print(f"Using device: {self.device}")

            # Switch CWD so relative paths in upstream code continue to work.
            original_cwd = os.getcwd()
            os.chdir(self.root)

            try:
                # Face detector - same as inference.py line 27-28
                self.app = Face_detect_crop(name="antelope", root="./insightface_func/models")
                self.app.prepare(ctx_id=ctx_id, det_thresh=0.6, det_size=(640, 640))

                # Generator - same as inference.py line 31-38
                G_path = os.path.join(self.root, "weights", "G_unet_2blocks.pth")
                if not os.path.exists(G_path):
                    raise FileNotFoundError(f"GHOST generator weights not found: {G_path}")
                self.G = AEI_Net("unet", num_blocks=2, c_id=512)
                self.G.eval()
                self.G.load_state_dict(torch.load(G_path, map_location=self.device))
                if self.use_cuda:
                    self.G = self.G.cuda()
                    self.G = self.G.half()
                else:
                    self.G = self.G.to(self.device)

                # ArcFace - same as inference.py line 41-44
                arcface_path = os.path.join(self.root, "arcface_model", "backbone.pth")
                if not os.path.exists(arcface_path):
                    raise FileNotFoundError(f"ArcFace weights not found: {arcface_path}")
                self.netArc = iresnet100(fp16=False)
                self.netArc.load_state_dict(torch.load(arcface_path, map_location=self.device))
                self.netArc = self.netArc.to(self.device)
                self.netArc.eval()

                # Landmark detector - same as inference.py line 47
                coord_path = os.path.join(self.root, "coordinate_reg", "model", "2d106det")
                self.handler = Handler(coord_path, 0, ctx_id=ctx_id, det_size=640)

                self.is_initialized = True
                mode_str = "GPU" if self.use_cuda else "CPU"
                print(f"GHOST model initialized ({mode_str})")

            finally:
                os.chdir(original_cwd)

        except Exception as exc:
            traceback.print_exc()
            print(f"GHOST initialization failed: {exc}")
            self.is_initialized = False

    def swap(self, source_bytes: bytes, target_bytes: bytes) -> np.ndarray:
        """
        Perform face swap following the same logic as inference.py main() function.
        """
        if not self.is_initialized:
            raise RuntimeError("GHOST service not initialized")

        original_cwd = os.getcwd()
        os.chdir(self.root)
        try:
            # Decode images
            source_img = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
            target_img = cv2.imdecode(np.frombuffer(target_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if source_img is None or target_img is None:
                raise ValueError("Failed to decode input images")

            # Get source face crop - same as inference.py line 70-73
            # crop_face returns [aligned_img], we take the first one and convert BGR->RGB
            source_crop = crop_face(source_img, self.app, self.crop_size)
            if not source_crop or len(source_crop) == 0:
                raise ValueError("No face detected in source image")
            source = [source_crop[0][:, :, ::-1]]  # List of RGB source crops

            # For image-to-image swap, full_frames is just the target image - same as inference.py line 82-83
            full_frames = [target_img]

            # Get target faces - same as inference.py line 88-89
            # When target_faces_paths is empty, use get_target to detect faces automatically
            target = get_target(full_frames, self.app, self.crop_size)
            if not target or len(target) == 0:
                raise ValueError("No face detected in target image")
            set_target = False  # Using auto-detected target, same as line 90

            # Run model inference - same as inference.py line 103-112
            final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(
                full_frames,
                source,
                target,
                self.netArc,
                self.G,
                self.app,
                set_target,
                similarity_th=0.15,
                crop_size=self.crop_size,
                BS=1  # Single image, batch size 1
            )

            # Get final image - same as inference.py line 128
            result = get_final_image(final_frames_list, crop_frames_list, full_frames[0], tfm_array_list, self.handler)
            
            return result

        finally:
            os.chdir(original_cwd)


def build_app(service: GhostModelService) -> FastAPI:
    app = FastAPI(title="sber-swap GHOST API", version="1.0.0")

    @app.get("/health")
    async def health():
        if service.is_initialized:
            return {"status": "ok"}
        return JSONResponse({"status": "error", "message": "model not initialized"}, status_code=503)

    @app.post("/swap")
    async def swap(
        source_image: UploadFile = File(..., description="Source face image"),
        target_image: UploadFile = File(..., description="Target image to receive face"),
    ):
        try:
            src_bytes = await source_image.read()
            tgt_bytes = await target_image.read()
            result_img = service.swap(src_bytes, tgt_bytes)
            encoded = encode_image_to_base64(result_img)
            mode = "GPU" if service.use_cuda else "CPU (slow)"
            return {"image": encoded, "note": f"GHOST remote backend ({mode})"}
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            traceback.print_exc()
            return JSONResponse({"error": str(exc)}, status_code=500)

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Serve sber-swap GHOST as HTTP API")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=9000, help="Port number")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def main():
    args = parse_args()
    service = GhostModelService(cpu=args.cpu)
    app = build_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
