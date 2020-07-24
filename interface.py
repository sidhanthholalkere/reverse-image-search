import embed_caption
from ImageDatabase import *
import show_img

caption = input("Enter phrase to search: ")

cap_emb = embed_caption.se_text(caption)
db = ImageDatabase()
db.load_database("imageDB.pkl")
top_img_ids = db.get_top_imgs(cap_emb)
show_img.display_topk(top_img_ids)
