from fasteasySD import fasteasySD as fesd

test = fesd.FastEasySD(device='cpu',use_fp16=False)

mode = "txt2img"

images = test.make(mode="txt2img",
                model_type="SD",model_path="Lykon/dreamshaper-7",
                lora_path=".",lora_name="chamcham_new_train_lora_2-000001.safetensors",
                prompt="sharp details, sharp focus, masterpiece, best quality, chamcham(twitch), hair bell, hair ribbon, multicolored hair, two-tone hair, 1girl, solo,",
                n_prompt="bad hand,text,watermark,low quality,medium quality,blurry,censored,wrinkles,deformed,mutated text,watermark,low quality,medium quality,blurry,censored,wrinkles,deformed,mutated",
                seed=0,steps=8,cfg=2,height=960,width=512,prompt_strength=0.4,num_images=1)

if mode == "txt2img":
            
    pil_images = test.return_PIL(images)

    test.save_PIL(pils=pil_images,save_name="./fesd")

elif mode == "img2img":
            
    test.i2i_batch_save(images_list=images,base_name="./fesd_i2i")