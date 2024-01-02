import streamlit as slt 
import math
# it seems streamlit reruns the whole script when an event occurs

if 'total' not in slt.session_state:
    slt.session_state.total=16
rows=4
def set_total(total):
    #slt.session_state.total=total
    slt.write(f"total={total}")
    slt.write(f"rows={int(math.sqrt(total))}")
slt.sidebar.write("hello")
slt.sidebar.selectbox("select here",["a","b","c"])
slt.session_state.total=slt.slider("Number of images",16,64,step=4)
slt.button("set 64",on_click=set_total,args=[64])
slt.button("set 16",on_click=set_total,args=[16])
#slt.button("click me",on_click=slt.write,args=[slt.session_state.total])
slt.button("Generate images",on_click=set_total,args=[slt.session_state.total])
slt.image("grid.png")


# import streamlit as st

# if 'button' not in st.session_state:
#     st.session_state.button = False

# def click_button():
#     st.session_state.button = not st.session_state.button

# st.button('Click me', on_click=click_button)

# if st.session_state.button:
#     # The message and nested widget will remain on the page
#     st.write('Button is on!')
#     st.slider('Select a value')
# else:
#     st.write('Button is off!')