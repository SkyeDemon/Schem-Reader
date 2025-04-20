import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import pytesseract
import fitz  # PyMuPDF
from tkinter import scrolledtext
from ultralytics import YOLO
import tempfile
import os
import sys
import io

class OCRApp:
    def __init__(self, master):
        self.master = master
        master.title("Scheme Reader")

        # Получаем путь к директории, где находится исполняемый файл
        if getattr(sys, 'frozen', False):
            # Если программа запущена как исполняемый файл
            self.base_path = os.path.dirname(sys.executable)
        else:
            # Если программа запущена из исходного кода
            self.base_path = os.path.dirname(__file__)

        # Указываем путь к модели YOLO
        self.yolo_model_path = os.path.join(self.base_path, 'yolo8n80ep.pt')
        self.yolo_model = None
        self.master = master
        master.title("Scheme Reader")

        self.image_path = None
        self.original_image = None
        self.rotated_image = None
        self.processed_image = None
        self.extracted_text = ""
        self.rotation_angle = 0  # Store current rotation angle


        # Кнопки и поле для ввода
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(pady=5)

        self.button_load = tk.Button(self.button_frame, text="Загрузить схему", command=self.load_image)
        self.button_load.pack(side=tk.LEFT, padx=5)

        self.button_process = tk.Button(self.button_frame, text="Распознать", command=self.process_image, state=tk.DISABLED)
        self.button_process.pack(side=tk.LEFT, padx=5)
        self.button_process.config(width=20, height=1)

        self.button_save_image = tk.Button(self.button_frame, text="Сохранить изображение", command=self.save_processed_image, state=tk.DISABLED)
        self.button_save_image.pack(side=tk.LEFT, padx=5)

        self.button_exit = tk.Button(self.button_frame, text="Выход", command=master.destroy)
        self.button_exit.pack(side=tk.LEFT, padx=5)

        # Поле для ввода и его заголовок
        self.input_label = tk.Label(self.button_frame, text="Введите текст для фильтра:")
        self.input_label.pack(side=tk.LEFT, padx=5)

        self.input_entry = tk.Entry(self.button_frame, width=30)  # Ширина поля
        self.input_entry.pack(side=tk.LEFT, padx=5)

        # Frame for rotation buttons
        self.rotate_frame = tk.Frame(master)
        self.rotate_frame.pack(pady=5)

        self.button_rotate_left = tk.Button(self.rotate_frame, text="<<<", command=lambda: self.rotate_image(-90))
        self.button_rotate_left.pack(side=tk.LEFT, padx=5)

        self.button_rotate_right = tk.Button(self.rotate_frame, text=">>>", command=lambda: self.rotate_image(90))
        self.button_rotate_right.pack(side=tk.LEFT, padx=5)

        # Поля для изменения размера области распознавания текста
        self.expand_x_label = tk.Label(self.button_frame, text="Размер по X:")
        self.expand_x_label.pack(side=tk.LEFT, padx=5)

        self.expand_x_entry = tk.Entry(self.button_frame, width=5)
        self.expand_x_entry.pack(side=tk.LEFT, padx=5)
        self.expand_x_entry.insert(0, "13")  # Изначальное значение для expand_x

        self.expand_y_label = tk.Label(self.button_frame, text="Размер по Y:")
        self.expand_y_label.pack(side=tk.LEFT, padx=5)

        self.expand_y_entry = tk.Entry(self.button_frame, width=5)
        self.expand_y_entry.pack(side=tk.LEFT, padx=5)
        self.expand_y_entry.insert(0, "35")  # Изначальное значение для expand_y


        # Создание вкладок
        self.tabControl = ttk.Notebook(master)
        self.tab1 = ttk.Frame(self.tabControl)
        self.tab2 = ttk.Frame(self.tabControl)
        self.tab3 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab1, text='Исходное изображение')
        self.tabControl.add(self.tab2, text='Распознанное изображение')
        self.tabControl.add(self.tab3, text='Текст')
        self.tabControl.pack(expand=1, fill="both")

        # Элементы для первой вкладки
        self.label_original = tk.Label(self.tab1)
        self.label_original.pack(expand=True, fill="both", pady=10)

        # Элементы для второй вкладки
        self.label_processed = tk.Label(self.tab2)
        self.label_processed.pack(expand=True, fill="both", pady=10)

        # Элементы для третьей вкладки
        self.text_area = scrolledtext.ScrolledText(self.tab3, wrap=tk.WORD)
        self.text_area.pack(expand=True, fill="both", padx=10, pady=10)

        # Привязка к событию выделения текста
        self.text_area.bind("<<Selection>>", self.copy_on_select)



    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.pdf"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            try:
                if file_path.lower().endswith(".pdf"):
                    pdf_document = fitz.open(file_path)
                    page = pdf_document[0]
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    self.original_image = Image.open(io.BytesIO(img_data))

                else:
                    self.original_image = Image.open(file_path)

                self.rotated_image = self.original_image.copy()  # Initialize rotated_image

                self.rotation_angle = 0  # Reset rotation angle

                self.display_image(self.rotated_image, self.label_original, 'original')
                self.processed_image = None
                self.display_image(self.processed_image, self.label_processed, 'processed')
                self.label_processed.config(image=None)
                self.label_processed.image = None
                self.text_area.config(state=tk.NORMAL)
                self.text_area.delete("1.0", tk.END)
                self.extracted_text = ""
                self.button_process.config(state=tk.NORMAL)
                self.button_save_image.config(state=tk.DISABLED)

                # Load YOLO model when image is loaded
                try:
                    self.yolo_model = YOLO(self.yolo_model_path)
                except Exception as e:
                    tk.messagebox.showerror("Ошибка", f"Не удалось загрузить YOLO модель: {e}")
                    self.yolo_model = None
            except Exception as e:
                tk.messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")
                self.image_path = None
                self.original_image = None
                self.button_process.config(state=tk.DISABLED)
                self.button_save_image.config(state=tk.DISABLED)


    def rotate_image(self, angle):
            if self.original_image:
                self.rotation_angle += angle
                self.rotated_image = self.original_image.rotate(self.rotation_angle, expand=True)
                self.display_image(self.rotated_image, self.label_original, 'original')
            else:
                tk.messagebox.showerror("Ошибка", "Сначала загрузите изображение.")

    def display_image(self, img, label, image_type):
        width = label.winfo_width()
        height = label.winfo_height()

        if width <= 0 or height <= 0:
            return

        if img:
            resized_image = img.copy()
            resized_image.thumbnail((width, height))
            photo = ImageTk.PhotoImage(resized_image)
            label.config(image=photo)
            label.image = photo
        else:
            label.config(image=None)
            label.image = None


    def process_image(self):
        if not self.image_path:
            tk.messagebox.showerror("Ошибка", "Сначала загрузите изображение.")
            return

        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                self.rotated_image.save(tmp_file.name)
                temp_image_path = tmp_file.name

            img = Image.open(temp_image_path)

            self.processed_image = img.copy()
            draw = ImageDraw.Draw(self.processed_image)
            font = ImageFont.truetype("arial.ttf", size=16)

            # Разрешенные символы
            allowed_chars = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789-,.\""

            # Получаем текст для фильтрации из поля ввода
            filter_text = self.input_entry.get().lower() #Получаем текст из поля ввода и переводим в нижний регистр

            # YOLO object detection BEFORE OCR
            yolo_boxes = []
            if self.yolo_model:
                try:
                    results = self.yolo_model.predict(img)
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            cls = int(box.cls[0])
                            # Проверка наличия класса в names
                            if cls < len(self.yolo_model.names) and self.yolo_model.names[cls] == 'device':
                                xyxy = box.xyxy[0].tolist()
                                conf = box.conf[0].item()
                                x1, y1, x2, y2 = map(int, xyxy)

                                draw.rectangle([(x1, y1), (x2, y2)], outline='green', width=3)
                                yolo_boxes.append((x1, y1, x2, y2))
                except Exception as e:
                    tk.messagebox.showerror("Ошибка", f"Ошибка при детекции объектов YOLO: {e}")

            extracted_text_list = []
            ocr_data = []

            if yolo_boxes:
                for x1, y1, x2, y2 in yolo_boxes:
                    try:
                        # Получаем значения из полей ввода и преобразуем их в целые числа
                        expand_x_factor = int(self.expand_x_entry.get())
                        expand_y_factor = int(self.expand_y_entry.get())

                        expand_x = int((x2 - x1) * (expand_x_factor / 100))  # Используем значение из поля для X
                        expand_y = int((y2 - y1) * (expand_y_factor / 100))  # Используем значение из поля для Y

                        expanded_x1 = max(0, x1 - expand_x)
                        expanded_y1 = max(0, y1 - expand_y)
                        expanded_x2 = min(img.width, x2 + expand_x)
                        expanded_y2 = min(img.height, y2 + expand_y)

                        cropped_img = img.crop((expanded_x1, expanded_y1, expanded_x2, expanded_y2))
                        scale_factor = 2
                        resized_width = int(cropped_img.width * scale_factor)
                        resized_height = int(cropped_img.height * scale_factor)
                        resized_img = cropped_img.resize((resized_width, resized_height), Image.LANCZOS)

                        # Добавляем параметр `charwhitelist` в pytesseract
                        data = pytesseract.image_to_data(resized_img, output_type=pytesseract.Output.DICT, lang='rus', config=f'-c tessedit_char_whitelist={allowed_chars}')
                        n_boxes = len(data['level'])

                        # Объединяем близкие символы
                        merged_boxes = []
                        current_box = None
                        x_tolerance = 10  # Adjust as needed

                        for i in range(n_boxes):
                            if data['text'][i].strip():
                                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                                text = data['text'][i]

                                if current_box is None:
                                    current_box = {'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h, 'text': text}
                                else:
                                    if abs(x - current_box['x2']) < x_tolerance:
                                        current_box['x2'] = x + w
                                        current_box['text'] += text
                                    else:
                                        merged_boxes.append(current_box)
                                        current_box = {'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h, 'text': text}

                        if current_box is not None:
                            merged_boxes.append(current_box)

                        # Фильтруем блоки текста по содержимому поля ввода
                        filtered_boxes = [box for box in merged_boxes if filter_text in box['text'].lower()]

                        # Transform coordinates and store
                        for box in filtered_boxes:
                            ocr_x = int(box['x1'] / scale_factor) + expanded_x1
                            ocr_y = int(box['y1'] / scale_factor) + expanded_y1
                            ocr_w = int((box['x2'] - box['x1']) / scale_factor)
                            ocr_h = int((box['y2'] - box['y1']) / scale_factor)
                            text_ocr = box['text']
                            ocr_data.append((ocr_x, ocr_y, ocr_x + ocr_w, ocr_y + ocr_h, text_ocr))


                        extracted_text = '\n'.join([box['text'] for box in filtered_boxes])
                        extracted_text_list.append(extracted_text)

                        draw.rectangle([(expanded_x1, expanded_y1), (expanded_x2, expanded_y2)], outline='blue', width=2)

                    except Exception as e:
                        print(f"OCR error within box: {e}")
                        extracted_text_list.append("")
            else: #If no YOLO boxes, do OCR on the whole image
                # Увеличение всего изображения перед распознаванием
                scale_factor = 2  # Можно настроить коэффициент увеличения
                resized_width = int(img.width * scale_factor)
                resized_height = int(img.height * scale_factor)
                resized_img = img.resize((resized_width, resized_height), Image.LANCZOS) #Image.Resampling.LANCZOS

                # Добавляем параметр `charwhitelist` в pytesseract
                data = pytesseract.image_to_data(resized_img, output_type=pytesseract.Output.DICT, lang='rus', config=f'-c tessedit_char_whitelist={allowed_chars}')
                n_boxes = len(data['level'])

                # Объединяем близкие символы
                merged_boxes = []
                current_box = None
                x_tolerance = 10  # Adjust as needed

                for i in range(n_boxes):
                    if data['text'][i].strip():
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        text = data['text'][i]

                        if current_box is None:
                            current_box = {'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h, 'text': text}
                        else:
                            if abs(x - current_box['x2']) < x_tolerance:
                                current_box['x2'] = x + w
                                current_box['text'] += text
                            else:
                                merged_boxes.append(current_box)
                                current_box = {'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h, 'text': text}

                if current_box is not None:
                    merged_boxes.append(current_box)
                # Фильтруем блоки текста по содержимому поля ввода
                filtered_boxes = [box for box in merged_boxes if filter_text in box['text'].lower()]

                extracted_text = '\n'.join([box['text'] for box in filtered_boxes])
                extracted_text_list.append(extracted_text)

                # Transform coordinates and store
                for box in filtered_boxes:
                    ocr_x = int(box['x1'] / scale_factor)
                    ocr_y = int(box['y1'] / scale_factor)
                    ocr_w = int((box['x2'] - box['x1']) / scale_factor)
                    ocr_h = int((box['y2'] - box['y1']) / scale_factor)
                    text_ocr = box['text']
                    ocr_data.append((ocr_x, ocr_y, ocr_x + ocr_w, ocr_y + ocr_h, text_ocr))

            self.extracted_text = "\n".join(extracted_text_list)

            # Draw OCR bounding boxes
            self.text_area.config(state=tk.NORMAL)
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert(tk.END, self.extracted_text)

            for x1, y1, x2, y2, text in ocr_data:
                draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=2)
                draw.text((x1, y2 + 5), text, fill='blue', font=font)

            self.display_image(self.processed_image, self.label_processed, 'processed')
            self.button_save_image.config(state=tk.NORMAL)

        except Exception as e:
            tk.messagebox.showerror("Ошибка", f"Ошибка распознавания текста: {e}")
            self.processed_image = None
            self.display_image(self.processed_image, self.label_processed, 'processed')
            self.extracted_text = ""
            self.button_save_image.config(state=tk.DISABLED)

        finally:
            if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
                os.remove(temp_image_path)


    def copy_on_select(self, event=None):
        try:
            selected_text = self.text_area.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.master.clipboard_clear()
            self.master.clipboard_append(selected_text)
            self.master.update()  # Ensure clipboard is updated
        except tk.TclError:
            pass  # Nothing is selected


    def save_processed_image(self):
        if self.processed_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".pdf",  # Изменен defaultextension
                                                     filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("PDF files", "*.pdf"), ("All files", "*.*")])  # Добавлен PDF
            if file_path:
                try:
                    if file_path.lower().endswith(".pdf"): #Если выбран PDF
                        img_byte_arr = io.BytesIO()
                        self.processed_image.save(img_byte_arr, format='PNG')  # Сохраняем во временный буфер в формате PNG
                        img_byte_arr = img_byte_arr.getvalue()
                        img_pil = Image.open(io.BytesIO(img_byte_arr))

                        doc = fitz.open()
                        page = doc.new_page(width=img_pil.width, height=img_pil.height)
                        page.insert_image(rect=page.rect, filename=None, stream=img_byte_arr) #Вставляем изображение из буфера

                        doc.save(file_path, garbage=4, deflate=True, clean=True)
                        doc.close()


                    else: #Если выбран другой формат, сохраняем как обычно
                        self.processed_image.save(file_path)

                    tk.messagebox.showinfo("Сохранение", "Изображение успешно сохранено.")
                except Exception as e:
                    tk.messagebox.showerror("Ошибка", f"Не удалось сохранить изображение: {e}")
        else:
            tk.messagebox.showerror("Ошибка", "Нет обработанного изображения для сохранения.")


root = tk.Tk()
root.geometry("1280x900")
app = OCRApp(root)
root.mainloop()