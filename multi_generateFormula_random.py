from transformers import BertTokenizerFast, RobertaForMaskedLM, GPT2LMHeadModel
import transformers
from tokenizers.implementations import BertWordPieceTokenizer

import os
import torch
import argparse
import pandas as pd
import random
from pymatgen.core.composition import Composition

import time
import threading

from queue import Queue



parser = argparse.ArgumentParser(description='Parent parser for tape functions',
                                     add_help=False)
parser.add_argument('--length', type=int, default=150, help='Base model class to run')

parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    
parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    
parser.add_argument("--k", type=int, default=0)

parser.add_argument("--p", type=float, default=0.9)

parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

parser.add_argument("--loop_num", type=int, default=1000, help="loop number")

parser.add_argument("--num_beam", type=int, default=1, help="beam number")

parser.add_argument("--max_length", type=int, default=256, help="max length of sentence")

parser.add_argument("--tokenizer", type=str, default=None, help="path of tokenizer") #/home/nihang/transformers/formulas_GPT2/icsd/trained/

parser.add_argument("--model_name", type=str, default=None, help="model name: GPT2LMHeadModel")

parser.add_argument("--model_path", type=str, default=None, help="path of trained model") # '/home/nihang/transformers/formulas_GPT2/icsd/trained/'

parser.add_argument("--save_path", type=str, default=None, help="path to save generated sequence") # '/home/nihang/transformers/formulas_GPT2/icsd/trained/'

parser.add_argument("--n_thread", type=int, default=10, help="the number of threads") # '/home/nihang/transformers/formulas_GPT2/icsd/trained/'
args = parser.parse_args()


# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer, max_len=512, do_lower_case=False)

# Load model
model_name = args.model_name
model = getattr(transformers, model_name).from_pretrained(args.model_path)

# Element list +
mapping_list = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",	"Al",	"Si",	"P",	"S",	"Cl",	"Ar",	"K", "Ca",	"Sc",	"Ti",	"V",	"Cr",	"Mn",	"Fe",	"Co",	"Ni",	"Cu",	"Zn",	"Ga",	"Ge",	"As",	"Se",	"Br",	"Kr",	"Rb",	"Sr",	"Y",	"Zr","Nb","Mo",	"Tc",	"Ru",	"Rh",	"Pd",	"Ag",	"Cd",	"In",	"Sn",	"Sb",	"Te",	"I",	"Xe",	"Cs",	"Ba",	"La",	"Ce",	"Pr",	"Nd",	"Pm",	"Sm",	"Eu",	"Gd",	"Tb",	"Dy",	"Ho",	"Er",	"Tm",  "Yb",	"Lu",	"Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",	"Hg",	"Tl",	"Pb",	"Bi",	"Po",	"At",	"Rn",	"Fr",	"Ra",	"Ac",	"Th",	"Pa",	"U",	"Np",	"Pu",	"Am",	"Cm",	"Bk",	"Cf",	"Es", "Fm",	"Md",	"No",	"Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg","Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
print("length of mapping list: ", len(mapping_list))

lock = threading.Lock()
def show(lk, q):
    generated_sequence = []
    time.sleep(1)
    input_str = mapping_list[random.randint(0,len(mapping_list)-1)] + " " + mapping_list[random.randint(0,len(mapping_list)-1)] + " " + mapping_list[random.randint(0,len(mapping_list)-1)] + " " + mapping_list[random.randint(0,len(mapping_list)-1)]  # generate started sequence (3 elements) randomly 
    
    #print(input_str)
    
    input_ids = torch.tensor(tokenizer.encode(input_str, add_special_tokens=True)).unsqueeze(0)
    #print(input_ids)
    
    length_i = len(input_str)

    output_sequences = model.generate(
        input_ids, 
        max_length=args.max_length, 
        num_beams=args.num_beam, 
        no_repeat_ngram_size=2, 
        num_return_sequences=1, 
        #early_stopping=True
        )
    
        
    special_token = ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]']

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        #print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
        generated_sequence = generated_sequence.tolist()
        text = tokenizer.decode(generated_sequence, skip_special_tokens=False, lowercase=False)
        #print(text)    
        for spec in special_token:
            text = text.replace(spec, "")
          
        q.put(text.strip()[length_i: ])
        
    #print(generated_sequence)    
    #q.put(generated_sequence)  
        

if __name__ == '__main__':
    
    #start_t = time.time() 
    q =Queue()
    threads = []
    generated_sequences = []
    for j in range(args.loop_num // args.n_thread):
        for i in range(args.n_thread):
            t = threading.Thread(target=show, args=(lock, q,))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()
        for _ in range(args.n_thread):    
            generated_sequences.append(q.get())  

    tmp_list = []
    for idx in range(len(generated_sequences)):
        tmp = generated_sequences[idx]
        x = tmp.split(".")
    
        i = 0
        for tmp_text in x:
            i += 1
            if (tmp_text != "") and (tmp_text != " ") and i!= 1:
                if (len(tmp_text.strip()) != 1) and (len(tmp_text.strip()) != 2):
                    tmp_list.append(tmp_text.strip())     ## generate splited sequence, but doesn't covnert to formulas
          
    tmp_list = list(set(tmp_list)) 


    formulas=[]
    for s in tmp_list:
        #print(s)
        if "<" in s:
            continue
        elements = set(s.split())
        if len(elements) ==1:
            continue
        if len(elements)>8:
            continue
        #print(elements)
        dict_pair={}
        for e in elements:
            dict_pair[e]=s.count(e)
            #print(dict_pair)
        if sum(dict_pair.values())>30:
            continue
        try:
            comp=Composition(dict_pair)
        except:
            continue
        formulas.append(comp.to_pretty_string())
            
    total_count = len(tmp_list)
    final_count = len(formulas)

    formulas = list(set(formulas)) 

    df1=pd.DataFrame(formulas)
    df1_col = ['pretty_formula']
    df1.columns = df1_col
    save_path = args.save_path
    df1.to_csv(os.path.join(save_path, 'generated_sequences.csv'),index=None)
    print("check formula_clean.csv file for results.")
    print('count b4 reduce=', total_count)
    print('final_count=',final_count)
    
    #print("Time: ", time.time()- start_t)    
        
