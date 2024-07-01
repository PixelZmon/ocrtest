import os
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter, A4
from tqdm import tqdm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage
from io import BytesIO
import re


def clean_text(text): # Función para limpiar y escapar el texto
    text = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ \t\n\r\f\v]', '', text)
    return text

#entrada y salida
input_pdf_path = r'C:\Users\sebastian\OneDrive - Universidad Católica de la Santísima Concepción\Escritorio\Entornos_py\OCR\ingenieria-ambiental.pdf'
output_pdf_path = r'C:\Users\sebastian\OneDrive - Universidad Católica de la Santísima Concepción\Escritorio\Entornos_py\OCR\libro_procesado_sin_imagenes.pdf'

if not os.path.exists(input_pdf_path): #Verificacion de existencia
    print(f"Error: El archivo {input_pdf_path} no existe.")
else:
    try:
        print("Comenzando la conversión del PDF a imágenes...")# convertir el PDF en imágenes solo si es que existe
        images = convert_from_path(input_pdf_path, 150)  # reducir la resolución a 150 DPI para reducir el uso de memoria
        print(f"PDF cargado correctamente. Número de páginas: {len(images)}")

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # ruta pytesseract

        
        doc = SimpleDocTemplate(output_pdf_path, pagesize=A4)#PDF DE SALIDA
        elements = []
        styles = getSampleStyleSheet()
                
        print("Comenzando a procesar las páginas...")#mostrar una barra de proceso en cmd
        for page_number, image in tqdm(enumerate(images), total=len(images), desc="Procesando páginas"):
            spa_image = Image.fromarray(np.array(image))# la imagen pasa a ser un objeto PIL para OCR
            ocr_data = pytesseract.image_to_data(spa_image, lang='spa', output_type=pytesseract.Output.DICT)# extracción de datos de la imagen

            gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY) # convertir la imagen a escala de grises 
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) #y aplicar umbral

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#detectar contornos

            for i in range(len(ocr_data['text'])):# intento de extraer y añadir el texto con posiciones precisas ajustadas a las de la imagen
                if int(ocr_data['conf'][i]) > 0:  # usar solo palabras con confianza > 0
                    x = int(ocr_data['left'][i] * A4[0] / spa_image.width)
                    y = A4[1] - int(ocr_data['top'][i] * A4[1] / spa_image.height) - int(ocr_data['height'][i] * A4[1] / spa_image.height)  # Ajustar la coordenada y
                    font_size = int(ocr_data['height'][i] * A4[1] / spa_image.height * 0.8)  # Ajustar el tamaño de fuente
                    text = clean_text(ocr_data['text'][i])
                    if text:  #asegurar que el texto no esté vacío después de la limpieza
                        elements.append(Paragraph(f'<font size={font_size}>{text}</font>', styles['Normal']))
            
            for cnt in contours: #intento de hacer que el codigo detecte y añada ecuaciones y gráficos
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h
                area = w * h

                if 0.8 < aspect_ratio < 1.2 and 50 < area < 500:#deteccion de ecuaciones con un área pequeña y un aspecto cercano a 1
                    roi = spa_image.crop((x, y, x + w, y + h))
                    eq_text = clean_text(pytesseract.image_to_string(roi, config='--psm 6'))
                    if eq_text:
                        elements.append(Paragraph(f'\\({eq_text}\\)', styles['Normal']))
                        elements.append(Spacer(1, 12))

                
                elif area >= 500:#criterios para detectar gráficos
                    roi = spa_image.crop((x, y, x + w, y + h))
                    # Redimensionar la imagen si es demasiado grande
                    max_width = A4[0] - 2 * 20  # márgenes de 20 puntos a cada lado
                    max_height = A4[1] - 2 * 20  # márgenes de 20 puntos arriba y abajo
                    width_ratio = max_width / w
                    height_ratio = max_height / h
                    scale_ratio = min(width_ratio, height_ratio, 1)
                    new_width = int(w * scale_ratio)
                    new_height = int(h * scale_ratio)
                    
                    buf = BytesIO()# convertir la imagen recortada a formato adecuado para la libreria ReportLab
                    roi = roi.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    roi.save(buf, format='PNG')
                    buf.seek(0)
                    rl_graphic = RLImage(buf, width=new_width, height=new_height)
                    elements.append(rl_graphic)
                    elements.append(Spacer(1, 12))

            elements.append(PageBreak())

        doc.build(elements)# construccion del pdf de salida final

        print(f"Texto extraído y guardado en {output_pdf_path}")
    except Exception as e:
        print(f"Error al procesar el PDF: {e}")
