{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ddfe39df-83ab-4cd4-b6e0-a937477c19d6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from huggingface_hub import login\n",
    "login(token=\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "36f9890c-b1c8-440f-b103-5338939ff4ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e12e765a-504d-46e6-b4b5-612ed8f934ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from diffusers import DiffusionPipeline\n",
    "pipe = DiffusionPipeline.from_pretrained(\"stabilityai/stable-diffusion-xl-base-1.0\", torch_dtype=torch.float16, use_safetensors=True, variant=\"fp16\")\n",
    "pipe.to(\"cuda\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3ef1d8b0-59be-4f54-96ca-4e39429b2d01",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"A futuristic city skyline at sunset\"\n",
    "\n",
    "images = pipe(prompt=prompt).images\n",
    "\n",
    "print(len(images))\n",
    "# for img in images:\n",
    "#     plt.imshow(img)\n",
    "#     plt.show()\n",
    "\n",
    "#images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d121fdf-3c34-4a25-bbae-52068dcfe389",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"A futuristic city skyline at sunset\"\n",
    "\n",
    "images = pipe(prompt=prompt).images[0]\n",
    "\n",
    "images\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26634fef-cf71-4478-8495-1ca8a287d593",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get the image dimensions (height, width, and optionally color channels)\n",
    "import numpy as np\n",
    "numpy_image = np.array(images)\n",
    "height, width, *channels = numpy_image.shape\n",
    "\n",
    "# Print the resolution\n",
    "print(f\"Image Resolution: {width}x{height} pixels\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "655b6b56-91a4-480d-accf-629dc058fda7",
   "metadata": {},
   "source": [
    "# Prompt Engineering: Improving Results"
   ]
  },
  {
   "cell_type": "raw",
   "id": "415511a8-6bac-451b-8f03-de6f7495ebc1",
   "metadata": {},
   "source": [
    "Vague prompt:\"cat\"\n",
    "Result: The model might generate a generic image of any type of cat, with unspecified style, color, environment, or mood.\n",
    ">> Vague prompts lead to unpredictable, generic results.\n",
    "\n",
    "\n",
    "Good prompt:\"A fluffy orange cat sitting on a windowsill, sunlight streaming through lace curtains, photorealistic\"\n",
    "Result: The image will likely include an orange, fluffy cat, sitting on a windowsill, with light effects from curtains, rendered in a photorealistic style.\n",
    ">> Good prompts with subject, details, style, color, and mood, lead to specific, high-quality images."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0891656f-28b3-4279-9691-56d66c8ec84c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Start with the subject\n",
    "prompt = \"A cat\"\n",
    "images = pipe(prompt=prompt).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ce68daa-07bb-4fd1-9cef-62c80d41279b",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"A realistic painting of a cat sitting by the window\"\n",
    "images = pipe(prompt=prompt).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d4c6b19-d4da-4c09-abc6-99412bdb8271",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"A realistic painting of a cat looking right and sitting by the window\"\n",
    "images = pipe(prompt=prompt).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fbc19296-2751-48ba-b825-952181babf13",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add style or details\n",
    "prompt = \"A realistic painting of a cat sitting by the window, watercolor\"\n",
    "images = pipe(prompt=prompt).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eea57aa2-6351-4a45-960a-31ddf1f82913",
   "metadata": {},
   "outputs": [],
   "source": [
    "#experimenting with styles (\"digital art\", \"watercolor\", \"photorealistic\", etc.)\n",
    "prompt = \"A realistic painting of a cat sitting by the window, digital art\"\n",
    "images = pipe(prompt=prompt).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d4089a8-fd7a-4314-86c3-f5b6462a7927",
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "TRY OUT THESE PROMPTS:\n",
    "\n",
    "prompt = \"portrait of a smiling girl\"\n",
    "prompt = \"portrait of a sad girl\"\n",
    "prompt = \"portrait of a angry girl\"\n",
    "prompt = \"portrait of a girl laughing\"\n",
    "'''"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7703f39-3900-4525-b0d5-ab1ed2d75c6b",
   "metadata": {},
   "source": [
    "# Parameters and Customization"
   ]
  },
  {
   "cell_type": "raw",
   "id": "e2462e67-1295-4d76-8cbb-e6f0ef7cbfea",
   "metadata": {},
   "source": [
    "Typical Adjustable Parameters:\n",
    "\n",
    ">> Seeds (seed): Control randomness. \n",
    ">> Steps (num_inference_steps): More steps, better detail (but slower).\n",
    ">> Resolution: Image size.\n",
    ">> Denoising Cutoff (denoising_end): Decides how far into the denoising process to go before stopping; a lower value means less denoising, so the result is closer to the starting noise."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4fefbb0a-dda7-4140-88bd-bfd3f1d93f68",
   "metadata": {},
   "source": [
    "## Seeds: Control randomness. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0e8f887b-8d78-4d90-b7ca-29a447b12a4b",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"A realistic painting of a cat sitting by the window\"\n",
    "images = pipe(prompt=prompt, seed=42).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "396a00e8-46a6-451d-9538-da78437c37e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"A realistic painting of a cat sitting by the window\"\n",
    "images = pipe(prompt=prompt, seed=42).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f317ec6e-2dbe-4645-a5fb-f0ab259b6e84",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"A realistic painting of a cat sitting by the window\"\n",
    "images = pipe(prompt=prompt, seed=42).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8e655ca8-afcc-428e-9e43-adfa06a87072",
   "metadata": {},
   "outputs": [],
   "source": [
    "# A better prompt alongside seed can be a good way to make results consistent but not exact\n",
    "prompt = \"A realistic painting of a cat looking right and sitting by the left side of window\"\n",
    "images = pipe(prompt=prompt, seed=42).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2cac6449-046f-4439-97c1-cdd555e23959",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"A realistic painting of a cat looking right and sitting by the left side of window\"\n",
    "images = pipe(prompt=prompt, seed=42).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13b59d36-3de9-436f-8223-090a6642c74d",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"A realistic painting of a cat looking right and sitting by the left side of window\"\n",
    "images = pipe(prompt=prompt, seed=314).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bee00817-385e-473f-af03-c8d9e6498c54",
   "metadata": {},
   "source": [
    "## Steps: More steps, better detail (but slower)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6be2e67d-330b-42dc-a30b-2d7814611084",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"A realistic portrait of a girl\"\n",
    "images = pipe(prompt=prompt, num_inference_steps=1).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0a18cc38-e131-4207-95c4-693e69164a07",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"A realistic portrait of a girl\"\n",
    "images = pipe(prompt=prompt, num_inference_steps=10).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b91b0c23-c7a8-402d-9894-c4dd9f2cffef",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"A realistic portrait of a girl\"\n",
    "images = pipe(prompt=prompt, num_inference_steps=30).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c00f9acd-8c86-4796-a974-786c54d9697f",
   "metadata": {},
   "source": [
    "# Resolution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "31937037-95da-4b6f-b1c6-233139e81a5a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set image resolution (width, height) - must be divisible by 8\n",
    "\n",
    "#width, height = 128, 128\n",
    "#width, height = 512, 512\n",
    "#width, height = 1024, 512\n",
    "#width, height = 512, 1024\n",
    "#width, height = 1024, 1024\n",
    "\n",
    "prompt = \"Potrait of a person eating apple\"\n",
    "images = pipe(prompt=prompt, width=width, height=height).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d3830cff-bf96-4bfb-89c1-03f5102eb4b4",
   "metadata": {},
   "source": [
    "# Logo Generation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f5ffe6b3-4297-4b36-958f-102d6b37d913",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"A tech startup logo with a sleek circuit pattern forming the letter ‘T,’ sharp lines, and a metallic gradient finish, modern and minimalist, vector style, white background\"\n",
    "images = pipe(prompt=prompt).images[0]\n",
    "images"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b792be73-1736-470e-8d1d-e586cef639d4",
   "metadata": {},
   "source": [
    "# Mixutre of Experts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d85fa587-3970-44d0-8d25-9b23597629d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from diffusers import DiffusionPipeline\n",
    "import torch\n",
    "\n",
    "# load both base & refiner\n",
    "base = DiffusionPipeline.from_pretrained(\n",
    "    \"stabilityai/stable-diffusion-xl-base-1.0\", torch_dtype=torch.float16, variant=\"fp16\", use_safetensors=True\n",
    ")\n",
    "\n",
    "base.to(\"cuda\")\n",
    "refiner = DiffusionPipeline.from_pretrained(\n",
    "    \"stabilityai/stable-diffusion-xl-refiner-1.0\",\n",
    "    text_encoder_2=base.text_encoder_2,\n",
    "    vae=base.vae,\n",
    "    torch_dtype=torch.float16,\n",
    "    use_safetensors=True,\n",
    "    variant=\"fp16\",\n",
    ")\n",
    "refiner.to(\"cuda\")\n",
    "\n",
    "# Define how many steps and what % of steps to be run on each experts (80/20) here\n",
    "n_steps = 40\n",
    "high_noise_frac = 0.8\n",
    "\n",
    "prompt = \"A majestic lion jumping from a big stone at night\"\n",
    "\n",
    "# run both experts\n",
    "image = base(\n",
    "    prompt=prompt,\n",
    "    num_inference_steps=n_steps,\n",
    "    denoising_end=high_noise_frac,\n",
    "    output_type=\"latent\",\n",
    ").images\n",
    "image = refiner(\n",
    "    prompt=prompt,\n",
    "    num_inference_steps=n_steps,\n",
    "    denoising_start=high_noise_frac,\n",
    "    image=image,\n",
    ").images[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b675ec63-bdc9-418e-8169-5813926359cf",
   "metadata": {},
   "source": [
    "# Limitations"
   ]
  },
  {
   "cell_type": "raw",
   "id": "a1cf602f-78e8-4dfd-82e9-44db77d7c14f",
   "metadata": {},
   "source": [
    ">The model does not achieve perfect photorealism\n",
    ">The model cannot render legible text\n",
    ">The model struggles with more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”\n",
    ">Faces and people in general may not be generated properly.\n",
    "The autoencoding part of the model is lossy."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "HAAI",
   "language": "python",
   "name": "haai"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
