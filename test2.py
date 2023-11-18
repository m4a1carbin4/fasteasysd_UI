from fasteasySD import FastEasySD as fesd

test = fesd.FastEasySD(device='cpu',use_fp16=False)

#~~~#

mode = "img2img"

images = test.make(mode=mode,seed=0,steps=4,prompt_strength=0.5,cfg=8,prompt="masterpeice, best quality, anime style",height=1063,width=827,num_images=2,input_image_dir="input.jpg")

if mode == "txt2img":
            
    pil_images = test.return_PIL(images)

    test.save_PIL(pils=pil_images,save_name="./fesd")

elif mode == "img2img":
            
    test.i2i_batch_save(images_list=images,base_name="./fesd_i2i")