import numpy as np
import gradio as gr
from lib import enhance_contact, enhance_contactless, hr_segmentation, segmentation
from lib.Fingerprint_Matching import infer
import os

fingerdict = {
    "Index": 0,
    "Middle":1,
    "Ring": 2,
    "Little": 3
}

def enhance(file, fg_type):
    if (fg_type == "contact-based"):
        enh = enhance_contact.main(file)
    else:
        enh = enhance_contactless.main(file)
    return enh[1]

def segment(file, hand):
    bounding_box, segments = segmentation.main(file, hand)
    return bounding_box

def match(file1, file2, match_type, ground_truth):
    if (match_type == "Contactless to Contactbased"):    
        enh, match = infer.main_cl2c(file1, file2)
    else: 
        enh, match = infer.main_cl2cl(file1, file2)
    return enh, match

def match_full(file1, file2, hand1, hand2, ground_truth):
    _, segments1 = segmentation.main(file1, hand1)
    _, segments2 = segmentation.main(file2, hand2)

    score, pred = infer.main_full_cl2cl(segments1, segments2)
    return score, pred


with gr.Blocks(title="RidgeBase Demo") as demo:
    gr.Markdown("Enhance, Segment, or Match")

    with gr.Tab("Segment"):
        with gr.Row():
            gr.Interface(fn=segment, 
            inputs=["image",gr.Dropdown(["RIGHT", "LEFT"])], 
            outputs=["image"],
            examples=[
                [os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_20294_1_RIGHT_image_fingerprintYPP5MU7E.png"), "RIGHT"],
                [os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_21583_1_RIGHT_image_fingerprint2A3FXT0T.png"), "RIGHT"],
                [os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_22837_2_LEFT_image_fingerprintY8VMUTA3.png"), "LEFT"],
                [os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_21583_1_LEFT_image_fingerprintWLQGVZCS.png"), "LEFT"],
                [os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_22657_1_LEFT_image_fingerprint44F2KEOY.png"), "LEFT"],
                [os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_22837_1_RIGHT_image_fingerprintNE5G7E5B.png"), "RIGHT"],
                [os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_28203_2_LEFT_image_fingerprintB4VVI7KR.png"), "LEFT"],
            ])

    with gr.Tab("Enhance"):
        with gr.Row():
            gr.Interface(
            fn=enhance, 
            inputs=["image",gr.Dropdown(["contactless", "contact-based"])], 
            outputs=["image"],
            examples=[
                [os.path.join(os.path.dirname(__file__), "./examples/contactbased/1_14493_Left_Index.bmp"), "contact-based"],
                [os.path.join(os.path.dirname(__file__), "./examples/contactbased/1_14493_Left_Middle.bmp"), "contact-based"],
                [os.path.join(os.path.dirname(__file__), "./examples/contactbased/1_14493_Left_Little.bmp"), "contact-based"],
                [os.path.join(os.path.dirname(__file__), "./examples/contactbased/1_20294_Left_Little.bmp"), "contact-based"],
                [os.path.join(os.path.dirname(__file__), "./examples/contactless/1_Apple_14493_1_LEFT_image_fingerprintSMEG5K05_0.png"), "contactless"],
                [os.path.join(os.path.dirname(__file__), "./examples/contactless/1_Apple_14493_1_LEFT_image_fingerprintSMEG5K05_1.png"), "contactless"],
                [os.path.join(os.path.dirname(__file__), "./examples/contactless/1_Apple_14493_1_LEFT_image_fingerprintSMEG5K05_2.png"), "contactless"],
                [os.path.join(os.path.dirname(__file__), "./examples/contactless/1_Apple_14493_1_LEFT_image_fingerprintSMEG5K05_3.png"), "contactless"],
            ])

    with gr.Tab("Match"):
        with gr.Row():
            gr.Interface(fn=match, 
            inputs=["image","image",gr.Dropdown(["Contactless to Contactbased", "Contactless to Contactless"]),"text"], 
            outputs=[
                gr.Textbox(label="Score"),
                gr.Textbox(label="Prediction Label"),
                ],
            examples=[ 
                #################### match
                [
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_12385_1_RIGHT_image_fingerprint3ECYBA38_0.9212412238121033_1.png"), 
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/12385/1_12385_Right_Middle.bmp"),
                "Contactless to Contactbased",
                "Match"
                ],
                [
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_10727_1_LEFT_image_fingerprint8GIFPDNU_0.9363613128662109_1.png"), 
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727/1_10727_Left_Middle.bmp"),
                "Contactless to Contactbased",
                "Match"
                ],
                [
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_10727_1_LEFT_image_fingerprint8GIFPDNU_0.9961502552032471_2.png"), 
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727/1_10727_Left_Ring.bmp"),
                "Contactless to Contactbased",
                "Match"
                ],
                #################### not match
                [
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_10727_1_LEFT_image_fingerprint8GIFPDNU_0.8888496160507202_3.png"), 
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/12385/1_12385_Left_Index.bmp"),
                "Contactless to Contactbased",
                "Not Match"
                ],
                [
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_10727_1_RIGHT_image_fingerprintDAH28AVI_0.8899842500686646_1.png"), 
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/12385/1_12385_Left_Ring.bmp"),
                "Contactless to Contactbased",
                "Not Match"
                ],
                #################### match
                [
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_10727_1_LEFT_image_fingerprint8GIFPDNU_0.9817909002304077_0.png"), 
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_10727_2_LEFT_image_fingerprint52FCXJQH_0.9899176955223083_0.png"),
                "Contactless to Contactless",
                "Match"
                ],
                [
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_10727_3_LEFT_image_fingerprintJ1GFM31O_0.993074893951416_2.png"), 
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_10727_1_LEFT_image_fingerprint8GIFPDNU_0.9961502552032471_2.png"),
                "Contactless to Contactless",
                "Match"
                ],
                #################### not match
                [
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_10727_1_LEFT_image_fingerprint8GIFPDNU_0.9961502552032471_2.png"), 
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_10727_2_LEFT_image_fingerprint52FCXJQH_0.9899176955223083_0.png"),
                "Contactless to Contactless",
                "Not Match"
                ],
                [
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_10727_3_LEFT_image_fingerprintJ1GFM31O_0.993074893951416_2.png"), 
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_11826_1_RIGHT_image_fingerprint2ZLGAB31_0.9904624819755554_0.png"),
                "Contactless to Contactless",
                "Not Match"
                ],
                [
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_10727_2_LEFT_image_fingerprint52FCXJQH_0.9899176955223083_0.png"), 
                os.path.join(os.path.dirname(__file__), "./examples/Fingerprints/10727_CL/1_Apple_12385_1_RIGHT_image_fingerprint3ECYBA38_0.9628813862800598_2.png"),
                "Contactless to Contactless",
                "Not Match"
                ],
            ])
   
    with gr.Tab("Match Full"):
        with gr.Row():
            gr.Interface(fn=match_full, 
            inputs=["image", "image", gr.Dropdown(["RIGHT", "LEFT"]), gr.Dropdown(["RIGHT", "LEFT"]), "text"], 
            outputs=[
                gr.Textbox(label="Score"),
                gr.Textbox(label="Prediction Label"),
                ],
            examples=[
                [os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_20294_1_LEFT_image_fingerprint4FMDN1C5.png"),
                 os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_20294_2_LEFT_image_fingerprint26HWHMCG.png"),
                 "LEFT",
                 "LEFT",
                 "Match"],

                [os.path.join(os.path.dirname(__file__), "./examples/1_Bhavin_1_RIGHT_image_fingerprintYZPQS6SA.png"),
                 os.path.join(os.path.dirname(__file__), "./examples/3_Bhavin_1_RIGHT_image_fingerprintT6AHIP0C.png"),
                 "RIGHT",
                 "RIGHT",
                 "Match"],

                [os.path.join(os.path.dirname(__file__), "./examples/1_Bhavin_1_LEFT_image_fingerprintT8UEBB67.png"),
                 os.path.join(os.path.dirname(__file__), "./examples/3_Bhavin_1_LEFT_image_fingerprintMDNLH0KU.png"),
                 "LEFT",
                 "LEFT",
                 "Match"],

                [os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_22657_1_RIGHT_image_fingerprintBTKT8N68.png"),
                 os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_22657_2_RIGHT_image_fingerprintPQPPGRQH.png"),
                 "RIGHT",
                 "RIGHT",
                 "Match"],
                 
                [os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_28203_1_LEFT_image_fingerprintP6R5RY7B.png"),
                 os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_21022_3_LEFT_image_fingerprintFY0N5WGT.png"),
                 "LEFT",
                 "LEFT",
                 "Not Match"],

                [os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_21583_1_RIGHT_image_fingerprint2A3FXT0T.png"),
                 os.path.join(os.path.dirname(__file__), "./examples/four_fingers/1_test_22657_1_RIGHT_image_fingerprint5N1AZL92.png"),
                 "RIGHT",
                 "RIGHT",
                 "Not Match"],
            ])

demo.launch(share=True)