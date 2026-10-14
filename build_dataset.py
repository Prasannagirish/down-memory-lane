"""Assemble dataset/ (image + .txt caption pairs) from training_photos/ and childhood_staging/."""
import os, shutil

ADULT = "a photo of girish as an adult man, "
adult = {
 "0DA949FD-7FCD-4B88-B26E-826D0A035F3A": "seen from above, wearing a light blue t-shirt and glasses, standing on a paved courtyard with a hand on his chin",
 "36BD788A-1569-4A5E-90EE-D9D2140DD10A": "wearing glasses and a dark denim shirt, smiling outdoors under green trees and a blue sky",
 "69F5CDB3-595C-4509-89F0-474542A28B47": "wearing round glasses and a white polka-dot shirt, indoors",
 "713194B7-C39C-4364-8323-66D834FDF84F": "wearing glasses and a light pink kurta with a flower garland, smiling",
 "81CBBBE1-B4EB-4027-BA8A-494583EDE132": "wearing a white t-shirt, selfie taken indoors, no glasses",
 "8B2A38B3-B1F4-4EBB-AC68-EF716F7D6F58": "wearing glasses and a light pink kurta with a flower garland, close-up portrait",
 "B87F95C0-21D0-4118-AEA4-BECE12D701DD": "wearing glasses and a dark navy sweater, selfie indoors",
 "COLOR_POP": "smiling selfie, wearing a maroon t-shirt, no glasses, in a bright room",
 "DB0986EB-833B-480B-9720-871C94E80D59": "wearing sunglasses, a dark blue shirt and a white dhoti, hand behind his head, standing outdoors on red soil",
 "F335C030-8C0C-49C4-A92E-DBCEDC5D7406": "wearing glasses and a blue t-shirt with a mustache, selfie in a corridor",
 "IMG-20230115-WA0030": "wearing a white t-shirt, slight smile, selfie against a yellow wall",
 "IMG20230327164943": "wearing glasses and a white t-shirt with blue stripes, selfie",
 "IMG_0888": "wearing glasses and a grey patterned shirt, looking down, outdoors at night with string lights",
 "IMG_1076": "wearing black rectangular glasses and a blue hooded jacket, selfie indoors",
 "IMG_1223": "wearing a yellow shirt and a white earbud, smirking, selfie indoors",
 "IMG_3430": "wearing an orange t-shirt, selfie indoors",
 "IMG_4966": "full body, wearing glasses, a light blue shirt, a navy tie and navy trousers, standing indoors",
 "d75c3b33-d330-470c-912e-5eb129af9ae2": "wearing glasses, a navy hoodie and an orange checked scarf, outdoors with a bridge and trees behind",
 "IMG_9115": "with a short beard, wearing a navy t-shirt, close-up selfie with a slight smile",
 "IMG_9443": "with a beard, wearing glasses, a cream polo shirt, a lanyard and a backpack, standing in front of a colourful illustrated wall",
 "IMG_9778": "with a beard, wearing glasses and a blue polo shirt, selfie outdoors with a large modern building and trees behind",
}
EXCLUDE = {"3B888FF7-4F3A-4E6B-8AF6-E83B6D85250E", "IMG_5841", "74D1F054-A0A0-4072-AD0A-AF9BBABAD11C"}  # weakest adult shots, replaced by the three newer beard photos
child = {  # staging file -> (age phrase, description)
 "child_01": ("a 7-year-old child", "wearing a grey suit and tie, smiling, standing in front of plants"),
 "child_02": ("a 10-year-old child", "selfie wearing a red and navy polo shirt, smiling"),
 "child_03": ("a 6-year-old child", "wearing a red shirt and a white dhoti, sitting on a wooden chair"),
 "child_04": ("an 8-year-old child", "wearing a red shirt, resting his chin on his hand"),
 "child_05": ("a 2-year-old toddler", "with a small black bindi on the forehead, sitting on a large toy leopard"),
 "child_09": ("a 10-year-old child", "selfie wearing a red and navy polo shirt"),
 "child_10": ("a 1-year-old baby", "with a small black bindi on the forehead, wearing headphones, sitting among soft blankets"),
}

TEEN = "teenage boy"
teen = {  # staging number -> (age phrase, description)
 2:  ("a 14-year-old teenage boy", "wearing a navy graphic t-shirt, selfie outdoors among trees"),
 5:  ("a 14-year-old teenage boy", "wearing blue over-ear headphones and a light blue t-shirt, selfie indoors"),
 7:  ("a 14-year-old teenage boy", "wearing a white graphic t-shirt, indoors in dim light"),
 6:  ("a 15-year-old teenage boy", "wearing a light blue t-shirt, selfie in a restaurant"),
 8:  ("a 15-year-old teenage boy", "wearing a cyan t-shirt, selfie outdoors beside a white wall"),
 10: ("a 16-year-old teenage boy", "wearing a light blue graphic t-shirt, holding a can of cola, outdoors at night on a street"),
 11: ("a 16-year-old teenage boy", "wearing a navy striped polo shirt with blue headphones around his neck, selfie indoors"),
 12: ("a 16-year-old teenage boy", "wearing a blue graphic t-shirt, holding a glass of orange juice, selfie indoors"),
 3:  ("a 17-year-old teenage boy", "full body, wearing a grey zip hoodie and black trousers, leaning against a wall, with a thin moustache"),
 4:  ("a 17-year-old teenage boy", "wearing a grey zip hoodie over a white graphic t-shirt, selfie in a corridor, with a thin moustache"),
 14: ("a 17-year-old teenage boy", "full body, wearing a blue patterned shirt and grey jeans, leaning on a red car outdoors"),
 15: ("an 18-year-old young man", "wearing a white graphic t-shirt, selfie outdoors at night, with a thin moustache"),
 17: ("a 19-year-old young man", "wearing a red graphic t-shirt, close-up selfie indoors, with a light moustache and stubble"),
}

shutil.rmtree("dataset", ignore_errors=True); os.makedirs("dataset")
def put(src, stem, caption):
    shutil.copy(src, f"dataset/{stem}.jpg"); open(f"dataset/{stem}.txt", "w").write(caption)
for f in sorted(os.listdir("training_photos")):
    if not f.endswith(".jpg"):
        continue
    stem = os.path.splitext(f)[0]
    if stem in EXCLUDE:
        continue
    put(f"training_photos/{f}", stem, ADULT + adult.pop(stem))
assert not adult, f"unused adult captions: {list(adult)}"
for stem, (age, desc) in child.items():
    put(f"childhood_staging/{stem}.jpg", stem, f"a photo of girish as {age}, {desc}")
for n, (age, desc) in teen.items():
    put(f"teen_staging/teen_{n:02d}.jpg", f"teen_{n:02d}", f"a photo of girish as {age}, {desc}")
print(len(os.listdir("dataset")) // 2, "image/caption pairs")
