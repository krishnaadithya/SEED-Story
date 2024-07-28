import gradio as gr
import hydra
from omegaconf import OmegaConf
import torch
import os
import re
from PIL import Image, ImageDraw
import json
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, EulerDiscreteScheduler

device = 'cuda:0'
dtype = torch.float16
dtype_str = 'fp16'
num_img_in_tokens = 64
num_img_out_tokens = 64
instruction_prompt = '{instruction}'

tokenizer_cfg_path = 'configs/tokenizer/clm_llama_tokenizer.yaml'
image_transform_cfg_path = 'configs/processer/qwen_448_transform.yaml'
visual_encoder_cfg_path = 'configs/visual_tokenizer/qwen_vitg_448.yaml'
llm_cfg_path = 'configs/clm_models/llama2chat7b_lora.yaml'
agent_cfg_path = 'configs/clm_models/agent_7b_sft.yaml'
adapter_cfg_path = 'configs/detokenizer/detokenizer_sdxl_qwen_vit_adapted.yaml'
discrete_model_cfg_path = 'configs/discrete_model/discrete_identity.yaml'
diffusion_model_path = 'pretrained/stable-diffusion-xl-base-1.0'
save_dir = "output"

tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg)
image_transform_cfg = OmegaConf.load(image_transform_cfg_path)
image_transform = hydra.utils.instantiate(image_transform_cfg)
visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
visual_encoder.eval().to(device, dtype=dtype)
llm_cfg = OmegaConf.load(llm_cfg_path)
llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype_str)
agent_model_cfg = OmegaConf.load(agent_cfg_path)
agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)
agent_model.eval().to(device, dtype=dtype)
noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").to(device, dtype=dtype)
unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet").to(device, dtype=dtype)
adapter_cfg = OmegaConf.load(adapter_cfg_path)
adapter = hydra.utils.instantiate(adapter_cfg, unet=unet).to(device, dtype=dtype).eval()
discrete_model_cfg = OmegaConf.load(discrete_model_cfg_path)
discrete_model = hydra.utils.instantiate(discrete_model_cfg).to(device).eval()

adapter.init_pipe(vae=vae, scheduler=noise_scheduler, visual_encoder=visual_encoder,
                  image_transform=image_transform, discrete_model=discrete_model, dtype=dtype, device=device)

boi_token_id = tokenizer.encode('<img>', add_special_tokens=False)[0]
eoi_token_id = tokenizer.encode('</img>', add_special_tokens=False)[0]

def add_subtitle(original_image, text):
    text_height = 80
    new_image_size = (original_image.width, original_image.height + text_height)
    new_image = Image.new("RGB", new_image_size, "black")
    new_image.paste(original_image, (0, 0))
    draw = ImageDraw.Draw(new_image)
    font_size = 14
    line1, line2 = text[:len(text) // 2], text[len(text) // 2:]
    text_position_line1 = (10, original_image.height + (text_height - font_size) // 2)
    text_color = "white"
    draw.text(text_position_line1, line1, fill=text_color)
    text_position_line2 = (10, text_position_line1[1] + font_size)
    draw.text(text_position_line2, line2, fill=text_color)
    return new_image

def process_image(image_path, question):
    image = Image.open(image_path).convert('RGB')
    init_image = add_subtitle(image, question)
    image_tensor = image_transform(image).unsqueeze(0).to(device, dtype=dtype)
    image_tokens = '<img>' + ''.join([f'<img_{i:05d}>' for i in range(num_img_in_tokens)]) + '</img>'
    prompt = instruction_prompt.format(instruction=question + image_tokens)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + input_ids
    boi_idx = input_ids.index(boi_token_id)
    eoi_idx = input_ids.index(eoi_token_id)
    input_ids = torch.tensor(input_ids).to(device, dtype=torch.long).unsqueeze(0)
    ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    ids_cmp_mask[0, boi_idx + 1:eoi_idx] = True
    embeds_cmp_mask = torch.tensor([True]).to(device, dtype=torch.bool)
    with torch.no_grad():
        image_embeds = visual_encoder(image_tensor)
    output = agent_model.generate(tokenizer=tokenizer, input_ids=input_ids, image_embeds=image_embeds,
                                  embeds_cmp_mask=embeds_cmp_mask, ids_cmp_mask=ids_cmp_mask, max_new_tokens=500,
                                  num_img_gen_tokens=num_img_out_tokens)
    text = re.sub(r'\s*<[^>]*>\s*', ' ', output['text']).strip()
    return init_image, text

def inference(image, question):
    image.save("temp_image.jpg")
    processed_image, response_text = process_image("temp_image.jpg", question)
    return processed_image, response_text

image_input = gr.inputs.Image(type="pil")
text_input = gr.inputs.Textbox(label="Question")
image_output = gr.outputs.Image(type="pil")
text_output = gr.outputs.Textbox(label="Generated Text")

gr_interface = gr.Interface(fn=inference, inputs=[image_input, text_input], outputs=[image_output, text_output])
gr_interface.launch()
