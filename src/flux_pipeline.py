import torch
import os
from diffusers import FluxKontextPipeline
from diffusers import BitsAndBytesConfig as DBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TBitsAndBytesConfig
from PIL import Image


QUANT_4BIT = True
QUANT_8BIT = False


class fluxPipeline():


    def __init__(self) -> None:
        
        if QUANT_4BIT:

            self.bnb_cfg_diffusers = DBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            self.bnb_cfg_transformers = TBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            self.quant_cfg = PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": self.bnb_cfg_diffusers,
                    "text_encoder_2": self.bnb_cfg_transformers,
                }
            )

        elif QUANT_8BIT:

            self.bnb_cfg_diffusers   = DBitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True, 
            )
            self.bnb_cfg_transformers = TBitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
            )

            self.quant_cfg = PipelineQuantizationConfig(
                quant_mapping={
                    "transformer":    self.bnb_cfg_diffusers,
                    "text_encoder_2": self.bnb_cfg_transformers,
                }
            )

        self.pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            quantization_config=self.quant_cfg, 
            torch_dtype=torch.bfloat16,
        )

        self.pipe.enable_model_cpu_offload()


    def generate_image(self, ref, ff_origin, gender, used_age_group):
        
        # in_dir = os.path.join(os.path.dirname(__file__), "input")
        # ref = Image.open((os.path.join(in_dir, "hfg_test.png"))).convert("RGB")

        # prompt = (
        #     f"The subject is a {gender} with {ff_origin} features in the age range of {used_age_group}"
        #     "Create a passport-style picture: same person as in the reference, front-facing, "
        #     "neutral expression, shoulders visible, pure white background, studio lighting."
        #     "ultra realistic style, avoid cartoonish aspect features"
        # )

        prompt = (
            f"Realisitic passport-style picture of a {gender} with {ff_origin} features in the age range of {used_age_group}) ,"
            " preserve facial features, ultra photorealistic, natural skin, real lighning, DSLR qualitiy, Canon 5D."
            " Isolate main character: passport-style pic, front-facing, in a new white background. Do not segment."
        )

        image = self.pipe(
            image=ref,
            prompt=prompt,
            guidance_scale=1.0).images[0]
        
        out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        image.save(os.path.join(out_dir, "cat_with_hat.png"))
        ref.save(os.path.join(out_dir, "ref_image.png"))