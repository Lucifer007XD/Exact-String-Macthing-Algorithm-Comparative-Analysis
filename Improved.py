

import time
import memory_profiler
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import fitz  # PyMuPDF
import docx
import csv
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')




def convert_to_text(file_path):
    _, file_extension = os.path.splitext(file_path)
    text = ""

    if file_extension.lower() == ".pdf":
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()

    elif file_extension.lower() == ".docx":
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

    elif file_extension.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as txt_file:
            text = txt_file.read()

    elif file_extension.lower() == ".csv":
        with open(file_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                text += ",".join(row) + "\n"

    else:
        print("Unsupported file format:", file_extension)
        return None

    return text.strip()





def preprocess_text(text):
   
    # Tokenize the text
    tokens = word_tokenize(text)
    
   
   
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into a string
    preprocessed_text = " ".join(filtered_tokens)
    
    return preprocessed_text





def batch_text(text, batch_size):
    # Split text into batches of specified size
    batches = [text[i:i+batch_size] for i in range(0, len(text), batch_size)]
    return batches





def preprocess_strong_suffix(shift, bpos, pat, m):
    i = m
    j = m + 1
    bpos[i] = j

    while i > 0:
        while j <= m and pat[i - 1] != pat[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i
            j = bpos[j]
        i -= 1
        j -= 1
        bpos[i] = j

def preprocess_case2(shift, bpos, m):
    j = bpos[0]
    for i in range(m + 1):
        if shift[i] == 0:
            shift[i] = j
        if i == j:
            j = bpos[j]

def bm_string_match(text, pattern):
    matches = []
    s = 0
    m = len(pattern)
    n = len(text)

    bpos = [0] * (m + 1)
    shift = [0] * (m + 1)

    preprocess_strong_suffix(shift, bpos, pattern, m)
    preprocess_case2(shift, bpos, m)

    while s <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1
        if j < 0:
            matches.append(s)
            s += shift[0]
        else:
            s += shift[j + 1]
    
    return matches





def compute_prefix_function(pattern):
    prefix = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j > 0 and pattern[j] != pattern[i]:
            j = prefix[j - 1]
        if pattern[j] == pattern[i]:
            j += 1
        prefix[i] = j
    return prefix

def kmp_string_match(text, pattern):
    prefix = compute_prefix_function(pattern)
    j = 0
    positions = []  # to store the positions of the pattern
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = prefix[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == len(pattern):
            positions.append(i - (j - 1))  # add the position to the list
            j = prefix[j - 1]  # prepare for the next match
    return positions  # return the list of positions





def rabinkarp_string_match(txt, pat, q=101):
    M = len(pat)
    N = len(txt)
    d = 256  # Number of characters in the input alphabet
    p = 0    # Hash value for pattern
    t = 0    # Hash value for text
    h = 1

    # Calculate the hash value of pattern and the first window of text
    for i in range(M):
        p = (d * p + ord(pat[i])) % q
        t = (d * t + ord(txt[i])) % q

    # Calculate h: pow(d, M-1) % q
    for i in range(M - 1):
        h = (h * d) % q

    indexes = []

    # Slide the pattern over the text
    for i in range(N - M + 1):
        # Check if hash values match, and then compare characters
        if p == t:
            match = True
            for j in range(M):
                if txt[i + j] != pat[j]:
                    match = False
                    break
            if match:
                indexes.append(i)
                

        # Calculate hash value for the next window of text
        if i < N - M:
            t = (d * (t - ord(txt[i]) * h) + ord(txt[i + M])) % q

            # Ensure positive hash value
            if t < 0:
                t += q

    return indexes





def brute_force(text, pattern):
    n = len(text)
    m = len(pattern)
    indices = []
    for i in range(n - m + 1):
        j = 0
        while j < m and text[i + j] == pattern[j]:
            j += 1
        if j == m:
            indices.append(i)  # pattern found at index i
    return indices 




def search_pattern_in_batch(batch, pattern, match):
    return match(batch, pattern)




def measure_accuracy(total_expected_patterns, total_found_patterns):
    if total_expected_patterns == 0:
        return 0.0
    return min(total_found_patterns / total_expected_patterns, 1.0)





def measure_performance(algorithm, text, pattern, expected):
    start_time = time.time()
    memory_usage_before = memory_usage()[0]
    
    # Run the algorithm
    matches = algorithm(text, pattern)
    
    end_time = time.time()
    memory_usage_after = memory_usage()[0]

    accuracy_score=measure_accuracy(expected,len(matches))
    
    time_taken = end_time - start_time
    memory_used = memory_usage_after - memory_usage_before
    
    return time_taken*1000, memory_used,accuracy_score




import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from functools import partial
import os

class StringMatchingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("String Matching GUI")

        self.file_path = tk.StringVar()
        self.pattern = tk.StringVar()
        self.algorithm = tk.StringVar(value="KMP")
        self.preprocess = tk.BooleanVar(value=True)

        self.create_widgets()

    def create_widgets(self):
        # File Selection
        file_frame = ttk.Frame(self.root)
        file_frame.pack(pady=10)

        ttk.Label(file_frame, text="Select File:").grid(row=0, column=0, padx=5, pady=5)
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path, width=40)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)


        ttk.Label(file_frame, text="Enter Pattern:").grid(row=1, column=0, padx=5, pady=5)
        self.pattern_entry = ttk.Entry(file_frame, textvariable=self.pattern, width=40)
        self.pattern_entry.grid(row=1, column=1, padx=5, pady=5)

        # Algorithm Selection
        algorithm_frame = ttk.Frame(self.root)
        algorithm_frame.pack(pady=10)

        ttk.Label(algorithm_frame, text="Select Algorithm:").grid(row=0, column=0, padx=5, pady=5)
        algorithm_options = ["KMP", "Boyer-Moore", "Rabin-Karp", "Brute Force"]
        self.algorithm_menu = ttk.OptionMenu(algorithm_frame, self.algorithm,algorithm_options[0], *algorithm_options)
        self.algorithm_menu.grid(row=0, column=1, padx=5, pady=5)

        # Preprocessing Checkbox
        preprocess_frame = ttk.Frame(self.root)
        preprocess_frame.pack(pady=10)

        self.preprocess_check = ttk.Checkbutton(preprocess_frame, text="Preprocess Text", variable=self.preprocess)
        self.preprocess_check.grid(row=0, column=0, padx=5, pady=5)

        # Submit Button
        submit_button = ttk.Button(self.root, text="Submit", command=self.process_matching)
        submit_button.pack(pady=10)

        # Output Text Area
        output_frame = ttk.Frame(self.root)
        output_frame.pack(pady=10)

        ttk.Label(output_frame, text="Matching Positions:").pack(pady=5)
        self.output_text = tk.Text(output_frame, width=50, height=10)
        self.output_text.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*")])
        if file_path:
            self.file_path.set(file_path)

    def preprocess_text(self, text):
        text=re.sub(r'[^\w\s]', '', text)
        # Tokenize the text
        tokens = text.split()
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
        # Join tokens back into a string
        preprocessed_text = " ".join(filtered_tokens)
        return preprocessed_text

    def process_matching(self):
        self.output_text.delete('1.0', tk.END)

        file_path = self.file_path.get()
        if not file_path or not os.path.exists(file_path):
            self.output_text.insert(tk.END, "Please select a valid file.")
            return

        pattern = self.pattern.get()
        if not pattern:
            self.output_text.insert(tk.END, "Please enter a pattern.")
            return

        algorithm = self.algorithm.get()

        
        text=convert_to_text(file_path)

        if self.preprocess.get():
            text = self.preprocess_text(text)
            pattern=pattern.lower()

        if algorithm == "KMP":
            matches = kmp_string_match(text, pattern)
        elif algorithm == "Boyer-Moore":
            matches = bm_string_match(text, pattern)
        elif algorithm == "Rabin-Karp":
            matches = rabinkarp_string_match(text, pattern)
        elif algorithm == "Brute Force":
            matches = brute_force(text, pattern)
        else:
            self.output_text.insert(tk.END, "Invalid algorithm selection.")
            return

        if matches:
            self.output_text.insert(tk.END, "Pattern found at positions:\n")
            for match in matches:
                self.output_text.insert(tk.END, f"{match}\n")
        else:
            self.output_text.insert(tk.END, "Pattern not found.")

if __name__ == "__main__":
    root = tk.Tk()
    app = StringMatchingGUI(root)
    root.mainloop()
