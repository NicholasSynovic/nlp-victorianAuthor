import pandas
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

# load the model from file
model = load_model("../models/veaaCNN.h5")

# example passage to classify, using line from old man and the sea
new_passage = [
    "it to what used to be at home and to the blowing of the wind on and feeling very sad and solitary i picture myself going up to bed among the unused rooms and sitting on my bed side crying for a comfortable word from i picture myself coming down stairs in the morning and looking through a long ghastly of a staircase window at the school bell hanging on the top of an with a above it and the time when it shall ring j and the rest to work which is only second in my apprehensions to the time when the man with the wooden leg shall the rusty gate to give admission to the awful mr i cannot think i was a very dangerous character in any of these aspects but in all of them i carried the same warning on my back mr never said much to me but he was never harsh to me i suppose we were company to each other without talking i forgot to mention that he would talk to himself sometimes and grin and his fist and grind his teeth and pull his hair in an unaccountable manner but he had these and at first they frightened me though i soon got used to them chapter vi i my circle oe i had led this life about a month when the man with the wooden leg began to stump about with a and a bucket of water from which i inferred that preparations were making to receive mr and the boys i was not mistaken for the came into the before long and turned out mr and me who lived where we could and got on how we could for some days during which we were always in the way of two or three young women who had rarely shown themselves before and were so continually in the midst of dust that i almost as much as if house had been a great snuff box one day i was informed by mr that mr would be home that evening in the evening after tea i heard that he was come before bed time i was fetched by the man with the wooden leg to appear before him mr s part of the house was a good deal more comfortable than ours and he had a snug bit of garden that looked pleasant after the dusty which was such a desert in miniature that i thought no one but a or a could have felt at home in it it seemed to me a bold thing even to take notice that the passage looked comfortable as i went on my way trembling to mr s presence which so the personal history and experience abashed me when i was ushered into it that i hardly saw mrs or miss who were both there in the parlor or anything but mr a stout gentleman with a bunch of watch chain and in an arm chair with a and bottle beside him so said mr this is the young gentleman whose teeth are to be filed turn him round the wooden legged man turned me about so as to exhibit the and having afforded time for a full survey of it turned me about again with my face to mr and posted himself at mr s side mr s face was fiery and his eyes were small and deep in his bead he had thick veins in his forehead a little nose and a large chin he was bald on the top of his head and had some thin wet looking hair that was just turning grey brushed across each temple so that the two sides on his forehead but the circumstance about him which impressed me most was that he had no voice but spoke in a whisper the exertion this cost him or the consciousness of talking in that feeble way made his angry face so much more angry and his thick veins so much thicker when he spoke that i am not surprised on looking back at this peculiarity striking me as his chief one now said mr what s the report of this boy there s nothing against him yet returned the man with the wooden leg there has been no opportunity i thought mr was disappointed i thought mrs and miss at whom i now glanced for the first time and who were both thin and quiet were not disappointed come here sir said mr to me come here said the man with the wooden leg repeating the gesture i have the happiness of knowing your father in law whispered mr taking me by the ear and a worthy man he is and a man of a strong character he knows me and i know him do you know me hey said mr my ear with ferocious not yet sir i said with the pain not yet hey repeated mr but you will soon hey you will soon hey repeated the man with the wooden leg i afterwards found that he generally acted with his strong voice as mr s to the boys i was very much frightened and said i hoped so if he pleased i felt all this while as if my ear were blazing he pinched it so hard i tell you what i am whispered mr letting it go at last with a screw at parting that brought the water into my eyes i m a a said the man with the wooden leg when i say i do a thing i do it said mr and when i say i will have a thing done i will have it done  will have a thing done i will have it done repeated the man with the wooden leg i am a determined character said mr that s what i am i do my duty that s what i do my flesh and blood  he"
]

# preprocess the passage
top_words = 100001
max_review_length = 1000
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(new_passage)
Sequence = tokenizer.texts_to_sequences(new_passage)
padded_sequence = sequence.pad_sequences(Sequence, maxlen=max_review_length)

# make prediction
author_classification = pandas.DataFrame(
    enumerate(model.predict(padded_sequence)[0]), columns=["Author", "Score (%)"]
)

# retrieve author, label mapping file
label_mapping = pandas.read_excel(
    "Author_Label_Mapping.xlsx", header=None, names=["key", "value"]
)
dictionary = label_mapping.set_index("key")["value"].to_dict()

# Format output
author_classification = author_classification.sort_values(
    by="Score (%)", ascending=False
)
author_classification["Author"] = (author_classification["Author"] + 1).map(dictionary)
author_classification["Score (%)"] = author_classification["Score (%)"] * 100

print(author_classification.head(5))

print("Correct Author: James Baldwin")
