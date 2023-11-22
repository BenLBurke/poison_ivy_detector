# %%
import gradio as gr
from fastai.vision.all import *
import timm
# %%
plant_type = ['Boxelder plant', 'Jack-in-the-Pulpit plant', 'posion_ivy plant', 'virgina creeper plant']
def classify_image(img):
    is_poison,_,probs = learn.predict(img)
    return dict(zip(plant_type, map(float,probs)))

# %%
learn = load_learner('model_4_24_23.pkl')

# %%
image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples= ['poison_ivy.jpg', 'virgina_creeper.jpg', 'box_elder.jpg','jack_in_the_pulpit.jpg']

intf = gr.Interface(fn=classify_image
                    , inputs=image
                    , outputs = label
                    , examples = examples
                    , title='Is it poison ivy?'
                    , theme = 'Ajaxon6255/Emerald_Isle'
                    , description = "Take a picture, or use an example below."
 )
intf.launch(inline=False)



