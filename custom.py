
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path

# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the font file
font_path = os.path.join(my_dir, 'arial.ttf')

from nodes import KSampler, VAEDecode, VAEDecodeTiled, PreviewImage
from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

from tsc_utils import *
from comfy import samplers
SCHEDULERS = samplers.KSampler.SCHEDULERS + ["AYS SD1", "AYS SDXL", "AYS SVD"]

########################################################################################################################
# TSC KSampler (Custom)
class TSC_KSampler_Custom:
    empty_image = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"model": ("MODEL",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "latent_image": ("LATENT",),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "preview_method": (["auto", "latent2rgb", "taesd", "vae_decoded_only", "none"],),
                     "vae_decode": (["true", "true (tiled)", "false"],),
                     },
                "optional": { "optional_vae": ("VAE",),
                              "script": ("SCRIPT",),
                              "guider": ("GUIDER",)},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE", "NOISE", "SAMPLER",)
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "IMAGE", "NOISE", "SAMPLER",)
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "Efficiency Nodes/Sampling"

    
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
               preview_method, vae_decode, denoise=1.0, prompt=None, extra_pnginfo=None, my_unique_id=None,
               optional_vae=(None,), script=None, add_noise=None, start_at_step=None, end_at_step=None,
               return_with_leftover_noise=None, sampler_type="regular", guider=None):

        # Rename the vae variable
        vae = optional_vae

        # If vae is not connected, disable vae decoding
        if vae == (None,) and vae_decode != "false":
            print(f"{warning('KSampler(Efficient) Warning:')} No vae input detected, proceeding as if vae_decode was false.\n")
            vae_decode = "false"

        # refiner_model = refiner_positive = refiner_negative = None

        #---------------------------------------------------------------------------------------------------------------
        def keys_exist_in_script(*keys):
            return any(key in script for key in keys) if script else False

        #---------------------------------------------------------------------------------------------------------------
        def vae_decode_latent(vae, samples, vae_decode):
            if isinstance(samples, dict) and "samples" in samples:
                samples_tensor = samples["samples"]
            elif isinstance(samples, torch.Tensor):
                samples_tensor = samples
            else:
                raise ValueError(f"Unexpected type for samples: {type(samples)}")
            
            if "tiled" in vae_decode:
                return VAEDecodeTiled().decode(vae, {"samples": samples_tensor}, 320)[0]
            else:
                return VAEDecode().decode(vae, {"samples": samples_tensor})[0]
        # ---------------------------------------------------------------------------------------------------------------
        def process_latent_image(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                denoise, sampler_type, add_noise, start_at_step, end_at_step, return_with_leftover_noise,
                                vae, vae_decode, preview_method, guider):

            # Store originals
            original_calculation = comfy.samplers.calculate_sigmas
            original_KSampler_SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS
            previous_preview_method = global_preview_method()
            original_prepare_noise = comfy.sample.prepare_noise
            original_KSampler = comfy.samplers.KSampler
            original_model_str = str(model)

            # monkey patch the sample function
            def calculate_sigmas(model_sampling, scheduler_name: str, steps):
                if scheduler_name.startswith("AYS"):
                    return AlignYourStepsScheduler().get_sigmas(scheduler_name.split(" ")[1], steps, denoise=1.0)[0]
                return original_calculation(model_sampling, scheduler_name, steps)


            comfy.samplers.KSampler.SCHEDULERS = SCHEDULERS
            comfy.samplers.calculate_sigmas = calculate_sigmas

            # Initialize output variables
            samples = images = gifs = preview = cnet_imgs = None

            try:
                # Change the global preview method (temporarily)
                set_preview_method(preview_method)

                rng_source = cfg_denoiser = add_seed_noise = m_seed = m_weight = None
                # ------------------------------------------------------------------------------------------------------
                # Store run parameters as strings. Load previous stored samples if all parameters match.
                latent_image_hash = tensor_to_hash(latent_image["samples"])
                positive_hash = tensor_to_hash(positive[0][0])
                negative_hash = tensor_to_hash(negative[0][0])

                parameters = [original_model_str, seed, steps, cfg, sampler_name, scheduler, positive_hash, negative_hash,
                              latent_image_hash, denoise, sampler_type, add_noise, start_at_step, end_at_step,
                              return_with_leftover_noise, rng_source, cfg_denoiser, add_seed_noise, m_seed, m_weight]

                # Convert all elements in parameters to strings, except for the hash variable checks
                parameters = [str(item) if not isinstance(item, type(latent_image_hash)) else item for item in parameters]

                # Load previous latent if all parameters match, else returns 'None'
                samples = load_ksampler_results("latent", my_unique_id, parameters)

                if samples is None: # clear stored images
                    store_ksampler_results("image", my_unique_id, None)
                    store_ksampler_results("cnet_img", my_unique_id, None)

                if samples is not None: # do not re-sample
                    images = load_ksampler_results("image", my_unique_id)

                # Sample the latent_image(s) using the Comfy KSampler nodes
                elif sampler_type == "regular":
                    def calculate_sigmas_denoise(model, scheduler_name, steps, denoise):
                        total_steps = steps
                        if denoise < 1.0:
                            if denoise <= 0.0:
                                return torch.FloatTensor([])
                            total_steps = int(steps/denoise)

                        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler_name, total_steps).cpu()
                        sigmas = sigmas[-(steps + 1):]
                        return sigmas

                    if guider is None:
                        samples = KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                                    latent_image, denoise=denoise)[0] if denoise>0 else latent_image
                    else:
                        noise = Noise_RandomNoise(seed).generate_noise(latent_image)
                        sigmas = calculate_sigmas_denoise(model, scheduler, steps, denoise)
                        sampler = comfy.samplers.sampler_object(sampler_name)
                        
                        latent = latent_image
                        latent_image = latent["samples"]
                        latent = latent.copy()
                        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
                        latent["samples"] = latent_image

                        noise_mask = None
                        if "noise_mask" in latent:
                            noise_mask = latent["noise_mask"]
                        
                        x0_output = {}
                        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
                        
                        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
                        samples = guider.sample(noise, latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
                        samples = samples.to(comfy.model_management.intermediate_device())

                        latent["samples"] = samples

                # Cache the first pass samples in the 'last_helds' dictionary "latent" if not xyplot
                if not any(keys_exist_in_script(key) for key in ["xyplot"]):
                    store_ksampler_results("latent", my_unique_id, samples, parameters)

                # Decode image if not yet decoded
                if "true" in vae_decode:
                    if images is None:
                        images = vae_decode_latent(vae, samples, vae_decode)
                        # Store decoded image as base image of no script is detected
                        if all(not keys_exist_in_script(key) for key in ["xyplot", "hiresfix", "tile", "anim"]):
                            store_ksampler_results("image", my_unique_id, images)

                # Define preview images
                if preview_method == "none" or (preview_method == "vae_decoded_only" and vae_decode == "false"):
                    preview = {"images": list()}
                elif images is not None:
                    preview = PreviewImage().save_images(images, prompt=prompt, extra_pnginfo=extra_pnginfo)["ui"]

                # Define a dummy output image
                if images is None and vae_decode == "false":
                    images = TSC_KSampler_Custom.empty_image

            finally:
                # Restore global changes
                set_preview_method(previous_preview_method)
                comfy.samplers.KSampler = original_KSampler
                comfy.sample.prepare_noise = original_prepare_noise
                comfy.samplers.calculate_sigmas = original_calculation
                comfy.samplers.KSampler.SCHEDULERS = original_KSampler_SCHEDULERS

            return samples, images, gifs, preview

        # ---------------------------------------------------------------------------------------------------------------
        # Clean globally stored objects of non-existant nodes
        globals_cleanup(prompt)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # If not XY Plotting
        if not keys_exist_in_script("xyplot"):

            # Process latent image
            samples, images, gifs, preview = process_latent_image(model, seed, steps, cfg, sampler_name, scheduler,
                                            positive, negative, latent_image, denoise, sampler_type, add_noise,
                                            start_at_step, end_at_step, return_with_leftover_noise, vae, vae_decode, preview_method, guider)

            result = (model, positive, negative, samples, vae, images, Noise_RandomNoise(seed),  comfy.samplers.sampler_object(sampler_name),)
                
            if preview is None:
                return {"result": result}
            else:
                return {"ui": preview, "result": result}

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # If XY Plot
        elif keys_exist_in_script("xyplot"):

            latent_list = []
            image_tensor_list = []
            image_pil_list = []

            # If no vae connected, throw errors
            if vae == (None,):
                print(f"{error('KSampler(Efficient) Error:')} VAE input must be connected in order to use the XY Plot script.")

                return {"ui": {"images": list()},
                        "result": (model, positive, negative, latent_image, vae, TSC_KSampler_Custom.empty_image,Noise_RandomNoise(seed), None,)}

            # If vae_decode is not set to true, print message that changing it to true
            if "true" not in vae_decode:
                print(f"{warning('KSampler(Efficient) Warning:')} VAE decoding must be set to \'true\'"
                    " for the XY Plot script, proceeding as if \'true\'.\n")

            # Split the 'samples' tensor
            samples_tensors = torch.split(latent_image['samples'], 1, dim=0)

            # Check if 'noise_mask' exists and split if it does
            if 'noise_mask' in latent_image:
                noise_mask_tensors = torch.split(latent_image['noise_mask'], 1, dim=0)
                latent_tensors = [{'samples': img, 'noise_mask': mask} for img, mask in
                                  zip(samples_tensors, noise_mask_tensors)]
            else:
                latent_tensors = [{'samples': img} for img in samples_tensors]

            # Set latent only to the first of the batch
            latent_image = latent_tensors[0]

            # Unpack script Tuple (X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, dependencies)
            X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, cache_models, xyplot_as_output_image,\
                xyplot_id, dependencies = script["xyplot"]

            #_______________________________________________________________________________________________________
            # The below section is used to check wether the XY_type is allowed for the Ksampler instance being used.
            # If not the correct type, this section will abort the xy plot script.

            samplers = {
                "regular": {
                    "disallowed": [
                        "AddNoise", 
                        "ReturnNoise", 
                        "StartStep", 
                        "EndStep", 
                        "RefineStep",
                        "Refiner", 
                        "Refiner On/Off", 
                        "AScore+", 
                        "AScore-",
                        "Scheduler",
                        "Checkpoint",
                        "VAE",
                        "LoRA Batch",
                        "LoRA Wt",
                        "LoRA MStr",
                        "LoRA CStr",
                        "ControlNetStrength",
                        "ControlNetStart%",
                        "ControlNetEnd%",
                        ],
                    "name": "KSampler (Efficient)"

                }
            }

            # Define disallowed XY_types for each ksampler type
            def get_ksampler_details(sampler_type):
                return samplers.get(sampler_type, {"disallowed": [], "name": ""})

            def suggest_ksampler(X_type, Y_type, current_sampler):
                for sampler, details in samplers.items():
                    if sampler != current_sampler and X_type not in details["disallowed"] and Y_type not in details["disallowed"]:
                        return details["name"]
                return "a different KSampler"

            # In your main function or code segment:
            details = get_ksampler_details(sampler_type)
            disallowed_XY_types = details["disallowed"]
            ksampler_name = details["name"]

            if X_type in disallowed_XY_types or Y_type in disallowed_XY_types:
                error_prefix = f"{error(f'{ksampler_name} Error:')}"

                failed_type = []
                if X_type in disallowed_XY_types:
                    failed_type.append(f"X_type: '{X_type}'")
                if Y_type in disallowed_XY_types:
                    failed_type.append(f"Y_type: '{Y_type}'")

                suggested_ksampler = suggest_ksampler(X_type, Y_type, sampler_type)

                print(f"{error_prefix} Invalid value for {' and '.join(failed_type)}. "
                    f"Use {suggested_ksampler} for this XY Plot type."
                    f"\nDisallowed XY_types for this KSampler are: {', '.join(disallowed_XY_types)}.")

                return {"ui": {"images": list()},
                    "result": (model, positive, negative, latent_image, vae, TSC_KSampler_Custom.empty_image,Noise_RandomNoise(seed),  comfy.samplers.sampler_object(suggested_ksampler),)}

            #_______________________________________________________________________________________________________
            # Printout XY Plot values to be processed
            def process_xy_for_print(value, replacement, type_):
                if type_ == "Seeds++ Batch" and isinstance(value, list):
                    return [v + seed for v in value]  # Add seed to every entry in the list

                elif type_ == "LoRA" and isinstance(value, list):
                    # Return only the first Tuple of each inner array
                    return [[(os.path.basename(v[0][0]),) + v[0][1:], "..."] if len(v) > 1
                            else [(os.path.basename(v[0][0]),) + v[0][1:]] for v in value]
                else:
                    return replacement if value is None else value
            # Determine the replacements based on X_type and Y_type
            replacement_X = scheduler if X_type == 'Sampler' else None
            replacement_Y = scheduler if Y_type == 'Sampler' else None

            # Process X_value and Y_value
            X_value_processed = process_xy_for_print(X_value, replacement_X, X_type)
            Y_value_processed = process_xy_for_print(Y_value, replacement_Y, Y_type)

            print(info("-" * 40))
            print(info('XY Plot Script Inputs:'))
            print(info(f"(X) {X_type}:"))
            for item in X_value_processed:
                print(info(f"    {item}"))
            print(info(f"(Y) {Y_type}:"))
            for item in Y_value_processed:
                print(info(f"    {item}"))
            print(info("-" * 40))

            #_______________________________________________________________________________________________________
            # Perform various initializations in this section

            # If not caching models, set to 1.
            if cache_models == "False":
                vae_cache = ckpt_cache = lora_cache = refn_cache = 1
            else:
                # Retrieve cache numbers
                vae_cache, ckpt_cache, lora_cache, refn_cache = get_cache_numbers("XY Plot")
            # Pack cache numbers in a tuple
            cache = lora_cache
            # Add seed to every entry in the list
            X_value = [v + seed for v in X_value] if "Seeds++ Batch" == X_type else X_value
            Y_value = [v + seed for v in Y_value] if "Seeds++ Batch" == Y_type else Y_value

            # Set lora_stack to None if one of types are LoRA
            if "LoRA" in X_type or "LoRA" in Y_type:
                lora_stack = None

            # Optimize image generation by prioritization:
            priority = [
                "LoRA"
            ]
            
            conditioners = {
                "Positive Prompt S/R",
                "Negative Prompt S/R",
                "AScore+",
                "AScore-",
                "Clip Skip",
                "Clip Skip (Refiner)",
                "ControlNetStrength",
                "ControlNetStart%",
                "ControlNetEnd%"
            }
            # Get priority values; return a high number if the type is not in priority list
            x_priority = priority.index(X_type) if X_type in priority else 999
            y_priority = priority.index(Y_type) if Y_type in priority else 999

            # Check if both are conditioners
            are_both_conditioners = X_type in conditioners and Y_type in conditioners
            # Determine whether to flip
            flip_xy = (y_priority < x_priority and not are_both_conditioners)

            # Perform the flip if necessary
            if flip_xy:
                X_type, Y_type = Y_type, X_type
                X_value, Y_value = Y_value, X_value

            #_______________________________________________________________________________________________________
            # The below code will clean from the cache any ckpt/vae/lora models it will not be reusing.
            # Note: Special LoRA types will not trigger cache: "LoRA Batch", "LoRA Wt", "LoRA MStr", "LoRA CStr"

            # Map the type names to the dictionaries
            dict_map = {"LoRA": []}

            # Create a list of tuples with types and values
            type_value_pairs = [(X_type, X_value.copy()), (Y_type, Y_value.copy())]

            # Iterate over type-value pairs
            for t, v in type_value_pairs:
                if t in dict_map:
                    # Flatten the list of lists of tuples if the type is "LoRA"
                    if t == "LoRA":
                        dict_map[t] = [item for sublist in v for item in sublist]

            lora_dict = [[t,] for t in dict_map.get("LoRA", [])] if dict_map.get("LoRA", []) else []

            # Construct refn_dict
            refn_dict = []

            # If both ckpt_dict and lora_dict are not empty, manipulate lora_dict as described
            if lora_dict:
                lora_dict = [(lora_stack) for lora_stack in lora_dict]

            # Avoid caching models accross both X and Y
            if X_type == "LoRA":
                refn_dict = []

            ### Print dict_arrays for debugging
            ###print(f"lora_dict={lora_dict}\nrefn_dict={refn_dict}")

            # Clean values that won't be reused
            clear_cache_by_exception(xyplot_id, lora_dict=lora_dict)

            ### Print loaded_objects for debugging
            ###print_loaded_objects_entries()

            #_______________________________________________________________________________________________________
            # Function that changes appropriate variables for next processed generations (also generates XY_labels)
            def define_variable(var_type, var, seed, lora_stack, var_label, num_label):

                # Define default max label size limit
                max_label_len = 42

                # If var_type is "Seeds++ Batch", generate text label
                if var_type == "Seeds++ Batch":
                    seed = var
                    text = f"Seed: {seed}"

                elif "LoRA" in var_type:
                    if not lora_stack:
                        lora_stack = var.copy()
                    else:
                        # Updating the first tuple of lora_stack
                        lora_stack[0] = tuple(v if v is not None else lora_stack[0][i] for i, v in enumerate(var[0]))

                    max_label_len = 50 + (12 * (len(lora_stack) - 1))
                    lora_name, lora_model_wt, lora_clip_wt = lora_stack[0]
                    lora_filename = os.path.splitext(os.path.basename(lora_name))[0]

                    if var_type == "LoRA":
                        if len(lora_stack) == 1:
                            lora_model_wt = format(float(lora_model_wt), ".2f").rstrip('0').rstrip('.')
                            lora_clip_wt = format(float(lora_clip_wt), ".2f").rstrip('0').rstrip('.')
                            lora_filename = lora_filename[:max_label_len - len(f"LoRA: ({lora_model_wt})")]
                            if lora_model_wt == lora_clip_wt:
                                text = f"LoRA: {lora_filename}({lora_model_wt})"
                            else:
                                text = f"LoRA: {lora_filename}({lora_model_wt},{lora_clip_wt})"
                        elif len(lora_stack) > 1:
                            lora_filenames = [os.path.splitext(os.path.basename(lora_name))[0] for lora_name, _, _ in
                                              lora_stack]
                            lora_details = [(format(float(lora_model_wt), ".2f").rstrip('0').rstrip('.'),
                                             format(float(lora_clip_wt), ".2f").rstrip('0').rstrip('.')) for
                                            _, lora_model_wt, lora_clip_wt in lora_stack]
                            non_name_length = sum(
                                len(f"({lora_details[i][0]},{lora_details[i][1]})") + 2 for i in range(len(lora_stack)))
                            available_space = max_label_len - non_name_length
                            max_name_length = available_space // len(lora_stack)
                            lora_filenames = [filename[:max_name_length] for filename in lora_filenames]
                            text_elements = [
                                f"{lora_filename}({lora_details[i][0]})" if lora_details[i][0] == lora_details[i][1]
                                else f"{lora_filename}({lora_details[i][0]},{lora_details[i][1]})" for i, lora_filename in
                                enumerate(lora_filenames)]
                            text = " ".join(text_elements)
                else: # No matching type found
                    text=""

                def truncate_texts(texts, num_label, max_label_len):
                    truncate_length = max(min(max(len(text) for text in texts), max_label_len), 24)

                    return [text if len(text) <= truncate_length else text[:truncate_length] + "..." for text in
                            texts]

                # Add the generated text to var_label if it's not full
                if len(var_label) < num_label:
                    var_label.append(text)

                # If var_type is LoRA, truncate entries in the var_label list when it's full
                if len(var_label) == num_label and "LoRA" in var_type:
                    var_label = truncate_texts(var_label, num_label, max_label_len)

                # Return the modified variables
                return lora_stack, var_label, seed

            #_______________________________________________________________________________________________________
            # The function below is used to optimally load Checkpoint/LoRA/VAE models between generations.
            def define_model(model, lora_stack, types, xyplot_id, cache):

                # Unpack types tuple
                X_type, Y_type = types

                # Note: Index is held at 0 when Y_type == "Nothing"

                # Load LoRA if required
                if (X_type == "LoRA"):
                    # Don't cache Checkpoints
                    model, _ = load_lora_flux(lora_stack, model, xyplot_id, cache=cache)
                elif Y_type == "LoRA":  # X_type must be Checkpoint, so cache those as defined
                    model, _ = load_lora_flux(lora_stack, model, xyplot_id, cache=None)


                return model

            # ______________________________________________________________________________________________________
            # The below function is used to generate the results based on all the processed variables
            def process_values(model,add_noise, seed, steps, start_at_step, end_at_step,
                               return_with_leftover_noise, cfg, sampler_name, scheduler, positive, negative,
                               latent_image, denoise, vae, vae_decode,
                               sampler_type, latent_list=[], image_tensor_list=[], image_pil_list=[], xy_capsule=None):

                capsule_result = None

                if capsule_result is None:
                    samples, images, _, _ = process_latent_image(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                                  latent_image, denoise, sampler_type, add_noise, start_at_step,
                                                  end_at_step, return_with_leftover_noise,
                                                  vae, vae_decode, preview_method, None)

                    # Add the latent tensor to the tensors list
                    latent_list.append(samples)

                    # Decode the latent tensor if required
                    image = images if images is not None else vae_decode_latent(vae, samples, vae_decode)

                    if xy_capsule is not None:
                        xy_capsule.set_result(image, samples)

                # Add the resulting image tensor to image_tensor_list
                image_tensor_list.append(image)

                # Convert the image from tensor to PIL Image and add it to the image_pil_list
                image_pil_list.append(tensor2pil(image))

                # Return the touched variables
                return latent_list, image_tensor_list, image_pil_list

            # ______________________________________________________________________________________________________
            # The below section is the heart of the XY Plot image generation

             # Initiate Plot label text variables X/Y_label
            X_label = []
            Y_label = []

            # Store types in a Tuple for easy function passing
            types = (X_type, Y_type)

            # Clone original model parameters
            def clone_or_none(*originals):
                cloned_items = []
                for original in originals:
                    try:
                        cloned_items.append(original.clone())
                    except (AttributeError, TypeError):
                        # If not clonable, just append the original item
                        cloned_items.append(original)
                return cloned_items
            
            original_model, original_positive, original_negative = clone_or_none(model, positive, negative)

            # Fill Plot Rows (X)
            for X_index, X in enumerate(X_value):
                # add a none value in the positive prompt memory.
                # the tuple is composed of (actual prompt, original prompte before S/R, prompt after X S/R)

                # Define X parameters and generate labels
                lora_stack, X_label, seed = \
                    define_variable(X_type, X, seed, lora_stack, X_label, len(X_value))
                
                if latent_list is None:
                    latent_list = []

                if X_type != "Nothing" and Y_type == "Nothing":

                    # Clone the model to retain the unpatched model
                    model_clone = model.clone()

                    # Models & Conditionings
                    model_clone = define_model(model_clone, lora_stack, types, xyplot_id, cache)

                    xy_capsule = None

                    # Generate Results
                    latent_list, image_tensor_list, image_pil_list = \
                        process_values(model_clone, add_noise, seed, steps, start_at_step, end_at_step,
                                       return_with_leftover_noise, cfg, sampler_name, scheduler, positive, negative,
                                       latent_image, denoise, vae, vae_decode, sampler_type, latent_list, image_tensor_list, image_pil_list, xy_capsule=xy_capsule)

                elif X_type != "Nothing" and Y_type != "Nothing":
                    for Y_index, Y in enumerate(Y_value):

                        # Define Y parameters and generate labels
                        lora_stack, Y_label, seed = \
                            define_variable(Y_type, Y, seed, lora_stack, Y_label, len(Y_value))
                        
                        # Clone the model to retain the unpatched model
                        model_clone = model.clone()

                        # Models & Conditionings
                        model_clone = define_model(model_clone, lora_stack, types, xyplot_id, cache)

                        # Generate Results
                        xy_capsule = None

                        
                        latent_list, image_tensor_list, image_pil_list = \
                            process_values(model_clone, add_noise, seed, steps, start_at_step, end_at_step,
                                           return_with_leftover_noise, cfg, sampler_name, scheduler[0],
                                           positive, negative, latent_image, denoise, vae, vae_decode,
                                           sampler_type, latent_list, image_tensor_list, image_pil_list, xy_capsule=xy_capsule)

            # Clean up cache
            if cache_models == "False":
                clear_cache_by_exception(xyplot_id, vae_dict=[], ckpt_dict=[], lora_dict=[], refn_dict=[])
            else:
                # Avoid caching models accross both X and Y
                if X_type == "LoRA":
                    clear_cache_by_exception(xyplot_id, ckpt_dict=[], refn_dict=[])

            # __________________________________________________________________________________________________________
            # Function for printing all plot variables (WARNING: This function is an absolute mess)
            def print_plot_variables(X_type, Y_type, X_value, Y_value, seed, scheduler, lora_stack,
                                     num_rows, num_cols, i_height, i_width):

                print("-" * 40)  # Print an empty line followed by a separator line
                print(f"{xyplot_message('XY Plot Results:')}")

                def get_lora_name(X_type, Y_type, X_value, Y_value, lora_stack=None):
                    lora_name = lora_wt = lora_model_str = lora_clip_str = None

                    # Check for all possible LoRA types
                    lora_types = ["LoRA"]

                    if X_type not in lora_types and Y_type not in lora_types:
                        if lora_stack:
                            names_list = []
                            for name, model_wt, clip_wt in lora_stack:
                                base_name = os.path.splitext(os.path.basename(name))[0]
                                formatted_str = f"{base_name}({round(model_wt, 3)},{round(clip_wt, 3)})"
                                names_list.append(formatted_str)
                            lora_name = f"[{', '.join(names_list)}]"
                    else:
                        if X_type in lora_types:
                            value = get_lora_sublist_name(X_type, X_value)
                            if  X_type == "LoRA":
                                lora_name = value
                                lora_model_str = None
                                lora_clip_str = None
                            if X_type == "LoRA Batch":
                                lora_name = value
                                lora_model_str = X_value[0][0][1] if lora_model_str is None else lora_model_str
                                lora_clip_str = X_value[0][0][2] if lora_clip_str is None else lora_clip_str
                            elif X_type == "LoRA MStr":
                                lora_name = os.path.basename(X_value[0][0][0]) if lora_name is None else lora_name
                                lora_model_str = value
                                lora_clip_str = X_value[0][0][2] if lora_clip_str is None else lora_clip_str
                            elif X_type == "LoRA CStr":
                                lora_name = os.path.basename(X_value[0][0][0]) if lora_name is None else lora_name
                                lora_model_str = X_value[0][0][1] if lora_model_str is None else lora_model_str
                                lora_clip_str = value
                            elif X_type == "LoRA Wt":
                                lora_name = os.path.basename(X_value[0][0][0]) if lora_name is None else lora_name
                                lora_wt = value

                        if Y_type in lora_types:
                            value = get_lora_sublist_name(Y_type, Y_value)
                            if  Y_type == "LoRA":
                                lora_name = value
                                lora_model_str = None
                                lora_clip_str = None
                            if Y_type == "LoRA Batch":
                                lora_name = value
                                lora_model_str = Y_value[0][0][1] if lora_model_str is None else lora_model_str
                                lora_clip_str = Y_value[0][0][2] if lora_clip_str is None else lora_clip_str
                            elif Y_type == "LoRA MStr":
                                lora_name = os.path.basename(Y_value[0][0][0]) if lora_name is None else lora_name
                                lora_model_str = value
                                lora_clip_str = Y_value[0][0][2] if lora_clip_str is None else lora_clip_str
                            elif Y_type == "LoRA CStr":
                                lora_name = os.path.basename(Y_value[0][0][0]) if lora_name is None else lora_name
                                lora_model_str = Y_value[0][0][1] if lora_model_str is None else lora_model_str
                                lora_clip_str = value
                            elif Y_type == "LoRA Wt":
                                lora_name = os.path.basename(Y_value[0][0][0]) if lora_name is None else lora_name
                                lora_wt = value

                    return lora_name, lora_wt, lora_model_str, lora_clip_str

                def get_lora_sublist_name(lora_type, lora_value):
                    if lora_type == "LoRA" or lora_type == "LoRA Batch":
                        formatted_sublists = []
                        for sublist in lora_value:
                            formatted_entries = []
                            for x in sublist:
                                base_name = os.path.splitext(os.path.basename(str(x[0])))[0]
                                formatted_str = f"{base_name}({round(x[1], 3)},{round(x[2], 3)})" if lora_type == "LoRA" else f"{base_name}"
                                formatted_entries.append(formatted_str)
                            formatted_sublists.append(f"{', '.join(formatted_entries)}")
                        return "\n      ".join(formatted_sublists)
                    else:
                        return ""

                # VAE, Checkpoint, Clip Skip, LoRA
                lora_name, lora_wt, lora_model_str, lora_clip_str = get_lora_name(X_type, Y_type, X_value, Y_value, lora_stack)

                # Seeds++ Batch
                seed = "\n      ".join(map(str, X_value)) if X_type == "Seeds++ Batch" else "\n      ".join(
                    map(str, Y_value)) if Y_type == "Seeds++ Batch" else seed

                scheduler = ", ".join([str(x[0]) if isinstance(x, tuple) else str(x) for x in X_value]) if X_type == "Scheduler" else \
                        ", ".join([str(y[0]) if isinstance(y, tuple) else str(y) for y in Y_value]) if Y_type == "Scheduler" else scheduler[0]


                #..........................................PRINTOUTS....................................................
                print(f"(X) {X_type}")
                print(f"(Y) {Y_type}")
                print(f"img_count: {len(X_value)*len(Y_value)}")
                print(f"img_dims: {i_height} x {i_width}")
                print(f"plot_dim: {num_cols} x {num_rows}")
                if lora_name:
                    print(f"lora: {lora_name}")
                print(f"seed: {seed}")

            # ______________________________________________________________________________________________________
            def adjusted_font_size(text, initial_font_size, i_width):
                font = ImageFont.truetype(str(Path(font_path)), initial_font_size)
                text_width = font.getlength(text)

                if text_width > (i_width * 0.9):
                    scaling_factor = 0.9  # A value less than 1 to shrink the font size more aggressively
                    new_font_size = int(initial_font_size * (i_width / text_width) * scaling_factor)
                else:
                    new_font_size = initial_font_size

                return new_font_size

            # ______________________________________________________________________________________________________

            def rearrange_list_A(arr, num_cols, num_rows):
                new_list = []
                for i in range(num_rows):
                    for j in range(num_cols):
                        index = j * num_rows + i
                        new_list.append(arr[index])
                return new_list

            def rearrange_list_B(arr, num_rows, num_cols):
                new_list = []
                for i in range(num_rows):
                    for j in range(num_cols):
                        index = i * num_cols + j
                        new_list.append(arr[index])
                return new_list

            # Extract plot dimensions
            num_rows = max(len(Y_value) if Y_value is not None else 0, 1)
            num_cols = max(len(X_value) if X_value is not None else 0, 1)

            # Flip X & Y results back if flipped earlier (for Checkpoint/LoRA For loop optimizations)
            if flip_xy == True:
                X_type, Y_type = Y_type, X_type
                X_value, Y_value = Y_value, X_value
                X_label, Y_label = Y_label, X_label
                num_rows, num_cols = num_cols, num_rows
                image_pil_list = rearrange_list_A(image_pil_list, num_rows, num_cols)
            else:
                image_pil_list = rearrange_list_B(image_pil_list, num_rows, num_cols)
                image_tensor_list = rearrange_list_A(image_tensor_list, num_cols, num_rows)
                latent_list = rearrange_list_A(latent_list, num_cols, num_rows)

            # Extract final image dimensions
            i_height, i_width = image_tensor_list[0].shape[1], image_tensor_list[0].shape[2]

            # Print XY Plot Results
            print_plot_variables(X_type, Y_type, X_value, Y_value, seed, scheduler, lora_stack,
                                 num_rows, num_cols, i_height, i_width)

            # Concatenate the 'samples' and 'noise_mask' tensors along the first dimension (dim=0)
            keys = latent_list[0].keys()
            result = {}
            for key in keys:
                tensors = [d[key] for d in latent_list]
                result[key] = torch.cat(tensors, dim=0)
            latent_list = result

            # Store latent_list as last latent
            ###update_value_by_id("latent", my_unique_id, latent_list)

            # Calculate the dimensions of the white background image
            border_size_top = i_width // 15

            # Longest Y-label length
            if len(Y_label) > 0:
                Y_label_longest = max(len(s) for s in Y_label)
            else:
                # Handle the case when the sequence is empty
                Y_label_longest = 0  # or any other appropriate value

            Y_label_scale = min(Y_label_longest + 4,24) / 24

            if Y_label_orientation == "Vertical":
                border_size_left = border_size_top
            else:  # Assuming Y_label_orientation is "Horizontal"
                # border_size_left is now min(i_width, i_height) plus 20% of the difference between the two
                border_size_left = min(i_width, i_height) + int(0.2 * abs(i_width - i_height))
                border_size_left = int(border_size_left * Y_label_scale)

            # Modify the border size, background width and x_offset initialization based on Y_type and Y_label_orientation
            if Y_type == "Nothing":
                bg_width = num_cols * i_width + (num_cols - 1) * grid_spacing
                x_offset_initial = 0
            else:
                if Y_label_orientation == "Vertical":
                    bg_width = num_cols * i_width + (num_cols - 1) * grid_spacing + 3 * border_size_left
                    x_offset_initial = border_size_left * 3
                else:  # Assuming Y_label_orientation is "Horizontal"
                    bg_width = num_cols * i_width + (num_cols - 1) * grid_spacing + border_size_left
                    x_offset_initial = border_size_left

            # Modify the background height based on X_type
            if X_type == "Nothing":
                bg_height = num_rows * i_height + (num_rows - 1) * grid_spacing
                y_offset = 0
            else:
                bg_height = num_rows * i_height + (num_rows - 1) * grid_spacing + 3 * border_size_top
                y_offset = border_size_top * 3

            # Create the white background image
            background = Image.new('RGBA', (int(bg_width), int(bg_height)), color=(255, 255, 255, 255))

            for row in range(num_rows):

                # Initialize the X_offset
                x_offset = x_offset_initial

                for col in range(num_cols):
                    # Calculate the index for image_pil_list
                    index = col * num_rows + row
                    img = image_pil_list[index]

                    # Paste the image
                    background.paste(img, (x_offset, y_offset))

                    if row == 0 and X_type != "Nothing":
                        # Assign text
                        text = X_label[col]

                        # Add the corresponding X_value as a label above the image
                        initial_font_size = int(48 * img.width / 512)
                        font_size = adjusted_font_size(text, initial_font_size, img.width)
                        label_height = int(font_size*1.5)

                        # Create a white background label image
                        label_bg = Image.new('RGBA', (img.width, label_height), color=(255, 255, 255, 0))
                        d = ImageDraw.Draw(label_bg)

                        # Create the font object
                        font = ImageFont.truetype(str(Path(font_path)), font_size)

                        # Calculate the text size and the starting position
                        _, _, text_width, text_height = d.textbbox([0,0], text, font=font)
                        text_x = (img.width - text_width) // 2
                        text_y = (label_height - text_height) // 2

                        # Add the text to the label image
                        d.text((text_x, text_y), text, fill='black', font=font)

                        # Calculate the available space between the top of the background and the top of the image
                        available_space = y_offset - label_height

                        # Calculate the new Y position for the label image
                        label_y = available_space // 2

                        # Paste the label image above the image on the background using alpha_composite()
                        background.alpha_composite(label_bg, (x_offset, label_y))

                    if col == 0 and Y_type != "Nothing":
                        # Assign text
                        text = Y_label[row]

                        # Add the corresponding Y_value as a label to the left of the image
                        if Y_label_orientation == "Vertical":
                            initial_font_size = int(48 * i_width / 512)  # Adjusting this to be same as X_label size
                            font_size = adjusted_font_size(text, initial_font_size, i_width)
                        else:  # Assuming Y_label_orientation is "Horizontal"
                            initial_font_size = int(48 *  (border_size_left/Y_label_scale) / 512)  # Adjusting this to be same as X_label size
                            font_size = adjusted_font_size(text, initial_font_size,  int(border_size_left/Y_label_scale))

                        # Create a white background label image
                        label_bg = Image.new('RGBA', (img.height, int(font_size*1.2)), color=(255, 255, 255, 0))
                        d = ImageDraw.Draw(label_bg)

                        # Create the font object
                        font = ImageFont.truetype(str(Path(font_path)), font_size)

                        # Calculate the text size and the starting position
                        _, _, text_width, text_height = d.textbbox([0,0], text, font=font)
                        text_x = (img.height - text_width) // 2
                        text_y = (font_size - text_height) // 2

                        # Add the text to the label image
                        d.text((text_x, text_y), text, fill='black', font=font)

                        # Rotate the label_bg 90 degrees counter-clockwise only if Y_label_orientation is "Vertical"
                        if Y_label_orientation == "Vertical":
                            label_bg = label_bg.rotate(90, expand=True)

                        # Calculate the available space between the left of the background and the left of the image
                        available_space = x_offset - label_bg.width

                        # Calculate the new X position for the label image
                        label_x = available_space // 2

                        # Calculate the Y position for the label image based on its orientation
                        if Y_label_orientation == "Vertical":
                            label_y = y_offset + (img.height - label_bg.height) // 2
                        else:  # Assuming Y_label_orientation is "Horizontal"
                            label_y = y_offset + img.height - (img.height - label_bg.height) // 2

                        # Paste the label image to the left of the image on the background using alpha_composite()
                        background.alpha_composite(label_bg, (label_x, label_y))

                    # Update the x_offset
                    x_offset += img.width + grid_spacing

                # Update the y_offset
                y_offset += img.height + grid_spacing

            xy_plot_image = pil2tensor(background)

         # Generate the preview_images
        preview_images = PreviewImage().save_images(xy_plot_image)["ui"]["images"]

        # Generate output_images
        output_images = torch.stack([tensor.squeeze() for tensor in image_tensor_list])

        # Set the output_image the same as plot image defined by 'xyplot_as_output_image'
        if xyplot_as_output_image == True:
            output_images = xy_plot_image

        # Print cache if set to true
        if cache_models == "True":
            print_loaded_objects_entries(xyplot_id, prompt)

        print("-" * 40)  # Print an empty line followed by a separator line

        result = (original_model, original_positive, original_negative, latent_list, optional_vae, output_images, Noise_RandomNoise(seed),  comfy.samplers.sampler_object(sampler_name),)
        return {"ui": {"images": preview_images}, "result": result}