import gradio as gr
from gradio.themes.utils import colors, fonts, sizes


pllava_theme = gr.themes.Monochrome(
    text_size="sm",
    spacing_size="sm",
    primary_hue=gr.themes.Color(c100="#f5f5f5", c200="#e5e5e5", c300="#d4d4d4", c400="#a3a3a3", c50="#fafafa", c500="#737373", c600="#525252", c700="#404040", c800="#262626", c900="#171717", c950="#000000"),
    secondary_hue=gr.themes.Color(c100="#f5f5f5", c200="#e5e5e5", c300="#d4d4d4", c400="#a3a3a3", c50="#fafafa", c500="#737373", c600="#525252", c700="#404040", c800="#262626", c900="#171717", c950="#000000"),
    neutral_hue=gr.themes.Color(c100="#f5f5f5", c200="#e5e5e5", c300="#d4d4d4", c400="#a3a3a3", c50="#fafafa", c500="#737373", c600="#525252", c700="#404040", c800="#262626", c900="#171717", c950="#000000"),
).set(
    background_fill_primary_dark='*primary_950',
    background_fill_secondary_dark='*neutral_950'
)

