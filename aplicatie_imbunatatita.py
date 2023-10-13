import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import time

def procesare(result): #se foloseste functie pentru resultatele obtinute in urma inferentei
    numar_mere = len(result.boxes)

    classes = result.names
    img = result.orig_img #imaginea originala

    for detection in result.boxes:  # atunci cand s au gasit detectii
        x1, y1, x2, y2 = detection.xyxy[0]  # extragem coordonatele dectiilor fiecarui mar din imagine, stocate intr un tensor Pytorch si sunt convertite in valori intregi pentru a putea fi procesate mai usor
        x1, y1, x2, y2 = x1.cpu().int().item(), y1.cpu().int().item(), x2.cpu().int().item(), y2.cpu().int().item()  # x1 y1 reprezinta punctul din stanga sus, iar x2 y2 reprezinta punctul din dreapta jos, metoda cpu este folosita pentru a converti tensorul din memoria GPU in memoria CPU,iar metoda .item() este folosita pentru a obtine valoarea std a tensoroului presupunand ca are un singur element

        imagine_rezultata = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255 ), 12)  # se deseneaza astfel un dreptunghi folosindu ma de coordonatele celor 2 puncte
        imagine_rezultata = cv2.putText(imagine_rezultata,f"{classes[detection.cls.item()]} {detection.conf.item():.2f}",(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10, cv2.LINE_AA)  # se obtine confidence ul pentru fiecare mar detectat

    return imagine_rezultata #se returneaza imaginea BGR

if "model" not in st.session_state: #var din ST se pastreaza chiar daca codul de reruleaza nu mi reincarca din nou modelul
    st.session_state.model = YOLO('best90acccolab.pt')
    st.session_state.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

st.title("Detecția fructelor")
st.subheader("Aveți de ales dintre detecția fructelor pe imagini sau pe videoclipuri")

testare_videoclip = st.checkbox("Testare pe videoclip")
if testare_videoclip:
    videoclip = st.file_uploader("Adaugati un videoclip", type = "Mp4")

    if videoclip:
        # Path relativ la aplicatia asta
        path_videoclip = f'./uploads/{videoclip.name}' #din memorie il scrie pe hard disk

        # Deschid fisierul ca scriere de bytes (documentatia la open)
        with open(path_videoclip, 'wb+') as f:
            f.write(videoclip.getvalue())
        container = st.empty()
        container_numar_mere = st.empty()
        for result in st.session_state.model.predict(source = path_videoclip, device=st.session_state.device,agnostic_nms = True, iou = 0.2, stream=True):
            imagine_rezultata = cv2.cvtColor(procesare(result), cv2.COLOR_BGR2RGB) #yolo foloseste BGR pe frame urile de la videoclipuri pentru ca foloseste OpenCV
            numar_mere = len(result.boxes)
            # Container e placeholder. Element gol si il schimb in imagine la fiecare frame
            container.image(imagine_rezultata)
            container_numar_mere.write(f"Sunt {numar_mere} mere.")


else:
    img_inserata = st.file_uploader("Incarca o imagine", type=["jpg", "jpeg", "png"])

    if img_inserata is not None:
        image = Image.open(img_inserata)
        st.image(image, caption="Imagine inserata" , use_column_width=True)

    if st.button("Submit"):
        if img_inserata is not None:

            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) #conversie de la RGB LA BGR

            results = st.session_state.model.predict(source = img, device=st.session_state.device,agnostic_nms = True, iou = 0.2)
            numar_mere = len(results[0].boxes)
            if numar_mere == 0:
                imagine_rezultata = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(imagine_rezultata, use_column_width=True)
                st.write(f"Nu sunt mere.")
            else:
                imagine_rezultata = procesare(results[0])
                imagine_rezultata = cv2.cvtColor(imagine_rezultata, cv2.COLOR_BGR2RGB) #se intoarce imagine BGR si trebuie converita la RGB
                st.image(imagine_rezultata, use_column_width=True)

                if numar_mere == 1:
                    st.write(f"Este un mar.")
                if numar_mere >= 2 and numar_mere < 20:
                    st.write(f"Sunt {numar_mere} mere.")
                else:
                    st.write(f"Sunt {numar_mere} de mere.")

                st.write("Detection results:")
                st.write(results)



