#!/usr/bin/env python
# coding: utf-8

# # Surface Charge Exchanger
# 
# This code was designed as part of my master thesis.
# About the project: 
# The object of study was an esterase that exhibited moderate stability in organic solvents. 
# A structurally similar esterase with much higher stability in organic solvents had a much more 
# negatively charged surface, with the object esterase showing a uniform charge distribution. 
# The hypothesis that changing the charge on the surface can increase stability was further investigated in 
# my master's thesis. The data of previous experiments was used to determine the cut-offs.

# By automating the data processing, the differences in output 
# when varying the input files could be quickly identified and 
# the optimal outcome was then tested in the lab.
#
# The main goal of this code was to obtain mutation suggestions 
# in order to modify the surface charge of the protein (an esterase)
# by exchanging positive amino acids with negative ones and *vice versa*. 
# 
# It checks, whether: 
# (1) the particular amino acid is conserved in the sequence so if 
#     its exchange would be critical for the protein function or folding process and 
# (2) the potential mutable amino acid is located on the surface.
# (3) the residue showed higher flexibility during a molecular dynamics simulation (RMSF) 
#     and thus would have fewer steric constraints to circumvent through the exchange.
# 
# If all three conditions are met, there would be mutations suggested based on:
# (1) structural similarity and
# (2) frequently occurance in multiple sequence alignment (MSA)
# Since this code was written for a specific esterase with a stable countrepart, 
# it also contains data from
# (3) the sequential alignment
# (4) structural alignment 
# to propose a more stable mutation.
#
# The mutations are suggested in form 
# "One letter code (Wild Type) - Position - One letter code (Mutatated)" (e.g. R23N)
# 
# If that turns out to be successful, you can use that with other proteins. 
# In this case, the part about structural and sequential alignment should be deleted or commented out.

import os
import numpy as np
import csv
import argparse
from math import sqrt
from tabulate import tabulate

### 1. Defining the Working Directory containing:

# Gromacs Output: "sasa_protein.xvg" and "rmsf.csv", 
# ConSurf Output: "msa_aa_variety_percentage.csv" and "consurf_grades.txt";

# If there is a related protein with desired features:
# PyMOL: Aligned PDB-files of protein of interest and the related protein (files with 11 and 12 columns are considered);
# PyMOL: Sequence Alignment (".aln")
# If no, please comment out (#) the alignment part and its results in ouput part

working_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(working_directory)

### 2. Defining Arguments

parser = argparse.ArgumentParser()

### 2.1. Parameter
parser.add_argument("-ch", help="Desired charge (positive or negative).", default="negative", type=str)
parser.add_argument("-ex", help="""Residues to exclude from mutation 
                    (one letter code separated by space)""", default=[], nargs="+", choices=["A","R","N","D","C","E","Q","G","H","I", 
           	    "L","K","M","F","P","S", "T","W","Y","V"])
parser.add_argument("-cs", help="Conservation score less than .. (Default 7)", default=7, type=int)
parser.add_argument("-rsasa", help="Relative solvent accessible surface area greater than .. % (Default 30)", default=30, type=float)
parser.add_argument("-rmsf", help="Root mean square fluctuation of residues greater than .. % (Default 0.15)", default=0.15, type=float)
parser.add_argument("-step", help="""'full' step to suggest mutation positive </> negative or polar > positive/negative,
		    'half' step to suggest mutations of charged residues towards polar.""", choices=["full", "half"], default="full", type=str)
		    
### 2.2. Input Data

parser.add_argument("-isasa", help="""Output file of Gromacs SASA analysis containing # and @ followed comments, used for orientation, followed by 3 columns: positions in the sequence, average absolute SASA values in nm/S2/N and its standard deviation.""", default="sasa_protein.xvg", type=str)
parser.add_argument("-pdbq", help="""PDB query: PDB-File of the protein of interest. 
                    If there is a related protein with desired features, 
                    please, provide the aligned data""", type=str)
parser.add_argument("-pdbap", help="""PDB alignment partner: PDB-File of related protein. 
                    with desired features, aligned to the protein of interest.""", type=str)
parser.add_argument("-aln", help="""Sequential alignment between the protein of interest 
                    and a related protein with desired features""", default="alignment.aln", type=str)

args = parser.parse_args()

### 3. Defining Lists and Variables

desired_charge = args.ch
half_or_full_step = args.step
excluded_aa = args.ex
cs_lt = args.cs
rsasa_gt = args.rsasa
rmsf_gt = args.rmsf

pdb_one = args.pdbq
pdb_two = args.pdbap
short_one = os.path.basename(os.path.splitext(pdb_one)[0])
short_two = os.path.basename(os.path.splitext(pdb_two)[0])
alignment_name = args.aln

print("\nParameter used: \nMutation Direction: " + str(desired_charge) + "\nConservation Score <= " + str(cs_lt) + "\nRelative Solvent Accessible Area >= " + str(rsasa_gt) + "%\nRoot Mean Square Fluctuation (nm) >= " + str(rmsf_gt) + "\nResidues to exclude: " + str([i for i in excluded_aa] if len(excluded_aa) != 0 else "-"))

# The empty lists created here will be filled with values after data import and conversion. 
# Each value with the index X corresponds to the residue with the position in the sequence X+1.

positions = []                    # position in the chain
amino_acids = []                  # one letter code
conservation_score = []           # 1 - variable, 9 - highly conserved

aSASA = []                        # absolute values for solvent accessible surface area
rSASA = []                        # relative values (rSASA) will be calculated later 
                                  # by dividing aSASA by maximum
                                  # rSASA > 30 %: exposed, <= 30%: buried
        
maxSASA = {"A": 129, "R": 274, "N": 195, "D": 193, "C": 167, 
           "E": 223, "Q": 225, "G": 104, "H": 224, "I": 197, 
           "L": 201, "K": 236, "M": 224, "F": 240, "P": 159, 
           "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174} 

three_letter_code = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q", 
                    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", 
                     "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
                     
mol_weight = {"A": 89, "R": 174, "N": 132, "D": 133, "C": 121, 
              "E": 147, "Q": 146, "G": 75, "H": 155, "I": 131, 
              "L": 131, "K": 146, "M": 149, "F": 165, "P": 115, 
              "S": 105, "T": 119, "W": 204, "Y": 181, "V": 117}

nonpolar = ["A", "F", "G", "H", "I", "L", "M", "P", "V", "W"]
polar = ["C", "N", "S", "T", "Q", "Y"]
positive = ["K", "R"]
negative = ["D", "E"]

charge = []                       # nonpolar, polar, positive or negative

size = []                         # molecular weight in Da

RMSF = []

mutability_general = []           # if residue meets the condition CS <= 3 and rSASA > 50 %
mutability = []                   # additional infos like residues to be excluded
                                  # or residues already have desired charge or nonpolar

mut_suggest_bo_size_frch = []     # mutation suggestion based on size (from oppositionally charged to polar or desired charge)      
mut_suggest_bo_size_fr0 = []      #  --- --- (from polar to desired charge)
mutation_list_bos = []            # combination of the two previous lists in form "wild type AA-position-mutated AA"

mut_suggest_bo_alignment_toch = [] # --- based on alignment (from oppositionally charged and polar to desired charge)
mut_suggest_bo_alignment_to0 = []  # --- --- (from oppositionally charged to polar)
mutation_list_boa = []             # combination of the two previous lists in form "wild type AA-position-mutated AA"
                                   # if both lists have a suggestion for same position, the most frequent one will be accepted 


### 4. Reding the File "sasa_protein.xvg" > Positions, aSASA, rSASA
# "sasa_protein.xvg" contains the positions of amino acids and 
# their mean solvent accessible surface area (aSASA) and the standard deviation of it.

sequence_length = 0  

sasa_start_key = '@ s1 legend "Standard deviation'

with open("sasa_protein.xvg", "r+") as sasa_file:
    
    sasa_all_lines = sasa_file.readlines()
    
    for count, line in enumerate(sasa_all_lines, 1):
        if sasa_start_key in line:
            #print("Table starts at line:", count+1)
            start_sasa = count
    
    sasa_table = sasa_all_lines[start_sasa:]
    
    for line in sasa_table:
        sequence_length += 1
        values = line.split()
        positions.append(int(values[0]))
        aSASA.append(float(values[1]))               # to enable the calculation with the list content
                                                     # it is necessary to convert the values into floats


### 5. Reading the File "consurf_grades.txt" > Amino Acids, Conservation Score

# "consurf_grades.txt" contains the amino acids with their positions in the sequence, 
# the conservation score and the predicted function for highly conserved residues 
# (s for structural, f for functional) as well as the variety of amino acids in the multiple sequence alignment etc.
     
consurf_start_key = "POS	 SEQ	SCORE"

with open("consurf_grades.txt") as consurf_file:
    whole_doc_lines = consurf_file.readlines()
    
    for count, line in enumerate(whole_doc_lines, 1):
        if consurf_start_key in line:
            #print("Table starts at line:", count + 2)
            start_consurf = count + 1
            
    consurf_table = whole_doc_lines[start_consurf:(sequence_length + start_consurf)]
    
    for line in consurf_table:
        values = line.split()
        
        cs_unform = values[3]                        # conservation scores are located in the 4th column
        cs_form = cs_unform.replace("*", "")         # the insufficient data are marked with an astrisk (*)
        conservation_score.append(int(cs_form))      # after removing the astrisk it is possible 
        amino_acids.append(values[1])                # to convert the data into integer 
                                                     # so that the data become comparable

for i in positions:
    rSASA.append(round(100 * float(100 * aSASA[i-1]/maxSASA[amino_acids[i-1]]), 2))
                            
### 6. Assigning the Residue Charge

def determine_charge(one_letter_code):
    if one_letter_code in nonpolar:
        return "nonpolar"
    elif one_letter_code in polar:
        return "polar"
    elif one_letter_code in positive:
        return "positive"
    elif one_letter_code in negative:
        return "negative"
    else:
        raise ValueError("Unknown amino acid: " + one_letter_code)

for i in positions:
    charge.append(determine_charge(amino_acids[i-1]))

def opposite_of(charge):
    if charge == "positive":
        return "negative"
    elif charge == "negative":
        return "positive"
    else:
        raise ValueError("Incorrect input ('positive' or 'negative' only)")

### 7. Import RMSF

with open("rmsf_mean.xvg") as rmsf_file:
    whole_doc_lines = rmsf_file.readlines()
    rmsf_table = whole_doc_lines[0:]
    for line in rmsf_table:
        values = line.split()
        RMSF.append(values[1])

### 8. Determine Mutability of Residues
# 
# Residues are classified as mutable if:
# 1) the conservation score is less than or equal to user defined value or 7 by default on a scale of 1 to 9;
# 2) the relative solvent accessible surface area is greater than user defined value or 30% by default;
# 3) the charge is either polar or the opposite of the desired charge;
# 4) they are not excluded from the mutation process by user in this step.  

for i in positions:
    if conservation_score[i-1] <= cs_lt and rSASA[i-1] > rsasa_gt  and float(RMSF[i-1]) >= rmsf_gt:
        mutability_general.append(True)
    else:
        mutability_general.append(False)
        
for i in positions:
    if (conservation_score[i-1] <= cs_lt and rSASA[i-1] > rsasa_gt and float(RMSF[i-1]) >= rmsf_gt 
        and amino_acids[i-1] not in excluded_aa and (charge[i-1] == opposite_of(desired_charge) 
        or charge[i-1] == "polar")):
        mutability.append(True)
    else:
        mutability.append(False)

### 9. Reading the File "msa_aa_variety_percentage.csv"
# 
# This csv table contains the residue variety in % for each position in the query sequence. 
# 
# In order not to produce unnecessary data volumes, only charged amino acids are considered. 
# 
# The data is imported in the form of csv, which are converted line by line 
# into dictionaries with column names as keys and the frequency in the alignment 
# in the concrete position as values. These dictionaries are then rewritten into 
# lists for every amino acid to be considered, and then dictionaries for specific 
# positions are generated in order to compare the frequency of occurrence.
# 
# To avoid disruptions the amino acids marked as to be excluded must only be removed 
# from these dictionaries after they have been generated.

aa_to_consider = []

aa_to_consider = polar + positive + negative

msa_start_key = "pos,A,C"

with open("msa_aa_variety_percentage.csv", newline = "") as residue_variety_file:
    
    msa_all_lines = residue_variety_file.readlines()
    
    for count, line in enumerate(msa_all_lines, 1):
        if msa_start_key in line:
            #print("Table starts at line:", count)
            msa_start = count - 1
            
    residue_variety_table = msa_all_lines[msa_start:]
    residue_variety_dict = csv.DictReader(residue_variety_table)
    
    for element in aa_to_consider:
            locals()[str(element)] = []
                     
    MAX_AA = [ line["MAX AA"] for line in residue_variety_dict ]
    
    for line in csv.DictReader(residue_variety_table):
        for key, value in line.items():
            if str(key) in aa_to_consider:
                if value == "":
                    locals()[str(key)].append(0)
                else:
                    locals()[str(key)].append(float(value)) 


### 10. Assigning the Residue Size to Enable Comparable Exchange
    
for amino_acid in amino_acids:
    size.append(mol_weight[amino_acid])

### 11. Defining Functions for Mutations in Positive Direction

def exchange_negative_with_positive(position):
    if mutability[position-1] == True and amino_acids[positions-1] in ["D","E"] and "K" not in excluded_aa:
        return "K"
    else:
        return "-"
        
def exchange_polar_with_positive(position):
    if mutability[position-1] == True and amino_acids[positions-1] in ["N","S","T"] and "R" not in excluded_aa:
        return "R"
    elif mutability[position-1] == True and amino_acids[positions-1] == "Q" and "K" not in excluded_aa:
        return "K"
    else:
        return "-"

def exchange_negative_with_polar(position):
    if mutability[position-1] == True and amino_acids[positions-1] == "D" and "N" not in excluded_aa:
        return "N"
    elif mutability[position-1] == True and amino_acids[positions-1] == "E" and "Q" not in excluded_aa:
        return "Q"
    else:
        return "-"

### 12. Defining Functions for Mutations in Negative Direction

def exchange_positive_with_negative(position):
    if mutability[position-1] == True and amino_acids[position-1] in ["R","K"] and "E" not in excluded_aa:
        return "E"
    else:
        return "-"
        
def exchange_positive_with_polar(position):
    if mutability[position-1] == True and amino_acids[position-1] in ["R","K"] and "Q" not in excluded_aa:
        return "Q"
    else:
        return "-"

def exchange_polar_with_negative(position):
    if mutability[position-1] == True and amino_acids[position-1] in ["N","S","T"] and "D" not in excluded_aa:
        return "D"
    elif mutability[position-1] == True and amino_acids[position-1] == "Q" and "E" not in excluded_aa:
        return "E"
    else: 
        return "-"


### 13. Defining Functions for Mutations from Alignment
# The functions return as a residue of certain charge (polar, negative or positive), that
# 1) does not belong to the previously determined residues to be excluded and
# 2) occurs most frequently in the alignment among residues of the same charge.

def align_exchange_charged_with_polar(position):
    polar_dict = {"C": C[position-1], "N": N[position-1], "S": S[position-1],
                  "T": T[position-1], "Q": Q[position-1], "Y": Y[position-1] }
    if mutability[position-1] == True and charge[position-1] == opposite_of(desired_charge):
        for i in excluded_aa:
            polar_dict.pop(i, None) 
        most_freq_polar = max(polar_dict, key = polar_dict.get)
        if any(x > 0 for x in polar_dict.values()): 
            return most_freq_polar
        else:
            return "-"
    else: 
        return "-"
    
    
def align_exchange_with_negative(position):
    neg_dict = {"D": D[position-1], "E": E[position-1] }
    if mutability[position-1] == True and (charge[position-1] == "polar" or charge[position-1] == "positive"):
        for i in excluded_aa:
            neg_dict.pop(i, None)  
        most_freq_neg = max(neg_dict, key = neg_dict.get)
        if any(x > 0 for x in neg_dict.values()):
            return most_freq_neg
        else:
            return "-"
    else: 
        return "-"

def align_exchange_with_positive(position):
    pos_dict = {"K": K[position-1], "R": R[position-1] }
    if mutability[position-1] == True and (charge[position-1] == "polar" or charge[position-1] == "negative"):
        for i in excluded_aa:
            pos_dict.pop(i, None)
        most_freq_pos = max(pos_dict, key = pos_dict.get)
        if any(x > 0 for x in pos_dict.values()):
            return most_freq_pos
        else:
            return "-"
    else: 
        return "-"
    
def get_the_best_align_suggest(position):
    if mut_suggest_bo_alignment_toch[position-1] == "-" and mut_suggest_bo_alignment_to0[position-1] == "-":
        return "-"
    elif mut_suggest_bo_alignment_toch[position-1] == "-" and mut_suggest_bo_alignment_to0[position-1] != "-":
        return mut_suggest_bo_alignment_to0[position-1]
    elif mut_suggest_bo_alignment_toch[position-1] != "-" and mut_suggest_bo_alignment_to0[position-1] == "-":
        return mut_suggest_bo_alignment_toch[position-1]
    elif mut_suggest_bo_alignment_toch[position-1] != "-" and mut_suggest_bo_alignment_to0[position-1] != "-":
        value1 = globals()[mut_suggest_bo_alignment_toch[position-1]][position-1]
        value2 = globals()[mut_suggest_bo_alignment_to0[position-1]][position-1]
        if value1 >= value2:
            return mut_suggest_bo_alignment_toch[position-1]
        elif value1 < value2:
            return mut_suggest_bo_alignment_to0[position-1]
        else:
            raise ValueError("Problem inside the 1st subcondition")
    else:
        raise ValueError("Problem inside the main condition")
            
### 14. Adjusting the Functions by Considering Desired Charge

full1 = locals()["exchange_" + str(opposite_of(desired_charge)) + "_with_" + str(desired_charge)]
full2 = locals()["exchange_polar_with_" + str(desired_charge)]
half = locals()["exchange_" + str(opposite_of(desired_charge)) + "_with_polar"]

alignment_exchange_charged = locals()["align_exchange_with_" + str(desired_charge)]

### 15. Getting Output as Lists

for i in positions:
    mut_suggest_bo_alignment_toch.append(alignment_exchange_charged(i)) 
    mut_suggest_bo_alignment_to0.append(align_exchange_charged_with_polar(i))
    
for i in positions:
    if get_the_best_align_suggest(i) != "-":
        mutation_list_boa.append(str(amino_acids[i-1] + str(positions[i-1]) + get_the_best_align_suggest(i)))
    elif get_the_best_align_suggest(i) == "-":
        mutation_list_boa.append("-")
    else: 
        raise ValueError("Weder noch?!")

for i in positions:       
    if half_or_full_step == "full":
        mut_suggest_bo_size_frch.append(full1(i))
        mut_suggest_bo_size_fr0.append(full2(i))
    elif half_or_full_step == "half":
        mut_suggest_bo_size_frch.append(half(i))
    elif mutability[i-1] == False:
        mut_suggest_bo_size_frch.append("-")
        mut_suggest_bo_size_fr0.append("-")
    else:
        raise ValueError("Etwas stimmt nicht! :(")

for i in positions:
    if mut_suggest_bo_size_frch[i-1] != "-":
        mutation_list_bos.append(str(str(amino_acids[i-1]) + str(positions[i-1]) + str(mut_suggest_bo_size_frch[i-1])))
    elif mut_suggest_bo_size_fr0[i-1] != "-":
        mutation_list_bos.append(str(str(amino_acids[i-1]) + str(positions[i-1]) + str(mut_suggest_bo_size_fr0[i-1])))
    else:
        mutation_list_bos.append("-")


# # Part II: Mutation Suggestions based on Alignment of ED30 with PT35

# II.1 Defining Lists

proteins = [short_one, short_two]

for protein in proteins:
    locals()[str(protein) + "_residues"] = []
    locals()[str(protein) + "_positions"] = []
    locals()[str(protein) + "_X"] = []
    locals()[str(protein) + "_Y"] = []
    locals()[str(protein) + "_Z"] = []
    locals()[str(protein) + "_pdb"] = str(protein) + ".pdb"    

# II.2. Getting Residues and their x, y, z - Coordinates from PDB Files

for protein in proteins:
    with open(str(protein) + ".pdb", "r+") as pdb_file:
        pdb_all_lines = pdb_file.readlines()
        for line in pdb_all_lines:
            values = line.split()
            if len(values) == 11 and values[2] == "CA":
                locals()[str(protein) + "_positions"].append(values[4])
                locals()[str(protein) + "_residues"].append(values[3])
                locals()[str(protein) + "_X"].append(float(values[5]))
                locals()[str(protein) + "_Y"].append(float(values[6]))
                locals()[str(protein) + "_Z"].append(float(values[7]))
            elif len(values) == 12 and values[2] == "CA":
            	locals()[str(protein) + "_positions"].append(values[5])
            	locals()[str(protein) + "_residues"].append(values[3])
            	locals()[str(protein) + "_X"].append(float(values[6]))
            	locals()[str(protein) + "_Y"].append(float(values[7]))
            	locals()[str(protein) + "_Z"].append(float(values[8]))
            else:
                pass  

coordinates_one = {"Positions": locals()[str(short_one) + "_positions"], "Residues": amino_acids, 
                   "Residues3LC": locals()[str(short_one) + "_residues"],
                   "X": locals()[str(short_one) + "_X"], "Y": locals()[str(short_one) + "_Y"], 
                   "Z": locals()[str(short_one) + "_Z"]}
                                 
#print(tabulate(coordinates_one, headers = "keys",  tablefmt = "fancy_grid"))

residues_two = [three_letter_code[i] for i in locals()[str(short_two) + "_residues"]]

coordinates_two = {"Positions": locals()[str(short_two) + "_positions"], 
                   "Residues": residues_two, "Residues3LC": locals()[str(short_two) + "_residues"],
                   "X": locals()[str(short_two) + "_X"], "Y": locals()[str(short_two) + "_Y"], 
                   "Z": locals()[str(short_two) + "_Z"]}
                                 
#print(tabulate(coordinates_two, headers = "keys",  tablefmt = "fancy_grid"))

### II.3. Sequence Alignment

ONE = []       
TWO = []
seq_alignment_mutations = []

with open(alignment_name, "r+") as alignment_file:
    alignment_all_lines = alignment_file.readlines()
    for line in alignment_all_lines:
        if str(short_one) in line:
            values = line.split()
            for letter in values[1]:
                ONE.append(letter)
        elif str(short_two) in line:
            values = line.split()
            for letter in values[1]:
                TWO.append(letter)
        else:
            pass
        
positions_ONE = []
positions_TWO = []
indices = []

indicis_count = 0
count_for_ONE = 0
count_for_TWO = 0

for position in ONE:
    if position != "-":
        count_for_ONE += 1
        indicis_count += 1
        positions_ONE.append(count_for_ONE)
        indices.append(indicis_count)
    elif position == "-":
        indicis_count += 1
        indices.append(indicis_count)
        positions_ONE.append("-")
    else:
        pass

for position in TWO:
    if position != "-":
        count_for_TWO += 1
        positions_TWO.append(count_for_TWO)
    elif position == "-":
        positions_TWO.append("-")
    else:
        pass

sequence_alignment = []

for index in indices:
    if ONE[index-1] != "-" and TWO[index-1] != "-":
        sequence_alignment.append(TWO[index-1])
    elif ONE[index-1] != "-" and TWO[index-1] == "-":
        sequence_alignment.append("-")
    else:
        pass

for i in positions:
    if (mutability[i-1] == True and charge[i-1] == opposite_of(desired_charge) and
        sequence_alignment[i-1] != "-" and
        (determine_charge(sequence_alignment[i-1]) == desired_charge or 
         determine_charge(sequence_alignment[i-1]) == "polar")):
        seq_alignment_mutations.append(sequence_alignment[i-1])
    elif (mutability[i-1] == True and charge[i-1] == "polar" and sequence_alignment[i-1] != "-" and
        determine_charge(sequence_alignment[i-1]) == desired_charge):
        seq_alignment_mutations.append(sequence_alignment[i-1])
    else:
        seq_alignment_mutations.append("-")

#print(seq_alignment_mutations)
mutation_list_boa_seq = []

for i in positions:
    if seq_alignment_mutations[i-1] != "-":
        mutation_list_boa_seq.append(str(amino_acids[i-1])+str(i) + str(seq_alignment_mutations[i-1]))
    else:
        mutation_list_boa_seq.append("-")
        
#print(mutation_list_boa_seq)

table_alignment = {"No1": positions_ONE, str(short_one): ONE, str(short_two): TWO, "No2": positions_TWO}
#print(tabulate(table_alignment, headers = "keys",  tablefmt = "fancy_grid"))

### II.4. Structural Alignment

alignment_mutations = []
distances = []
mutation_list_boa_str = []

for position in positions:    
    if mutability[position-1] == True:
        x = locals()[str(short_one) + "_X"][position-1]
        y = locals()[str(short_one) + "_Y"][position-1]
        z = locals()[str(short_one) + "_Z"][position-1]  
        charge_one = charge[position-1]   
        for residue in locals()[str(short_two) + "_positions"]:
            index = int(residue) - int(locals()[str(short_two) + "_positions"][0])
            x_two = locals()[str(short_two) + "_X"][index]
            y_two = locals()[str(short_two) + "_Y"][index]
            z_two = locals()[str(short_two) + "_Z"][index]
            distance = sqrt((x - x_two)**2 + (y - y_two)**2 + (z - z_two)**2)
            charge_two = determine_charge(residues_two[index])
            pot_mut = (str(position) + str(amino_acids[position-1]) + 
                   "-" + str(residue) + str(residues_two[index]))
            mutante = (str(amino_acids[position-1]) + str(position) + str(residues_two[index]))
            if distance <= 5: 
                if charge_one == opposite_of(desired_charge) and charge_two == desired_charge:
                    alignment_mutations.append(pot_mut)
                    mutation_list_boa_str.append(mutante)
                elif charge_one == opposite_of(desired_charge) and charge_two == "polar":
                    alignment_mutations.append(pot_mut)
                    mutation_list_boa_str.append(mutante)
                elif charge_one == "polar" and charge_two == desired_charge:
                    alignment_mutations.append(pot_mut)
                    mutation_list_boa_str.append(mutante) 
    else: 
        pass
    if len(mutation_list_boa_str) == position: 
        pass
    else: 
        mutation_list_boa_str.append("-")
              
#print(mutation_list_boa_str)

### III Output

# If there is no second protein, delete elements in tables 
# which are based on structural and sequence alignment ("Aln_best", "Aln_seq")

### III.1. Creating an Output Table

table_end = {"No": [i for i in positions if mutability[i-1] == True], 
             "R": [amino_acids[i-1] for i in positions if mutability[i-1] == True], 
             "CS": [conservation_score[i-1] for i in positions if mutability[i-1] == True], 
             "rSASA%": [rSASA[i-1] for i in positions if mutability[i-1] == True], 
             "RMSF": [RMSF[i-1] for i in positions if mutability[i-1] == True],
            "mut": [mutability[i-1] for i in positions if mutability[i-1] == True], 
             "charge": [charge[i-1] for i in positions if mutability[i-1] == True], 
             "size": [mutation_list_bos[i-1] for i in positions if mutability[i-1] == True], 
             "align_I": [mutation_list_boa[i-1] for i in positions if mutability[i-1] == True], 
             "align_ESeq": [mutation_list_boa_seq[i-1] for i in positions if mutability[i-1] == True], 
             "align_EStr": [mutation_list_boa_str[i-1] for i in positions if mutability[i-1] == True]}

#print(tabulate(table_end, headers = "keys",  tablefmt = "fancy_grid"))

if desired_charge == "negative":
    table = {"No": [i for i in positions if mutability[i-1] == True], 
         "AA": [amino_acids[i-1] for i in positions if mutability[i-1] == True], 
         "CS": [conservation_score[i-1] for i in positions if mutability[i-1] == True], 
         "rSASA%": [rSASA[i-1] for i in positions if mutability[i-1] == True], 
         "RMSF": [RMSF[i-1] for i in positions if mutability[i-1] == True],
         "D": [D[i-1]for i in positions if mutability[i-1] == True], 
         "E": [E[i-1]for i in positions if mutability[i-1] == True],  
         "C": [C[i-1]for i in positions if mutability[i-1] == True], 
         "N": [N[i-1]for i in positions if mutability[i-1] == True], 
         "S": [S[i-1]for i in positions if mutability[i-1] == True], 
         "T": [T[i-1]for i in positions if mutability[i-1] == True], 
         "Q": [Q[i-1]for i in positions if mutability[i-1] == True], 
         "Y": [Y[i-1]for i in positions if mutability[i-1] == True], 
         "Aln_best": [mutation_list_boa[i-1] for i in positions if mutability[i-1] == True],
         "Aln_seq": [mutation_list_boa_seq[i-1] for i in positions if mutability[i-1] == True], 
         "Aln_str": [mutation_list_boa_str[i-1] for i in positions if mutability[i-1] == True]}
    
elif desired_charge == "positive":
    table = {"No": [i for i in positions if mutability[i-1] == True], 
         "AA": [amino_acids[i-1] for i in positions if mutability[i-1] == True], 
         "CS": [conservation_score[i-1] for i in positions if mutability[i-1] == True], 
         "rSASA%": [rSASA[i-1] for i in positions if mutability[i-1] == True],
         "RMSF": [RMSF[i-1] for i in positions if mutability[i-1] == True],
         "R": [R[i-1]for i in positions if mutability[i-1] == True], 
         "K": [K[i-1]for i in positions if mutability[i-1] == True],
         "H": [H[i-1]for i in positions if mutability[i-1] == True],
         "C": [C[i-1]for i in positions if mutability[i-1] == True], 
         "N": [N[i-1]for i in positions if mutability[i-1] == True], 
         "S": [S[i-1]for i in positions if mutability[i-1] == True], 
         "T": [T[i-1]for i in positions if mutability[i-1] == True], 
         "Q": [Q[i-1]for i in positions if mutability[i-1] == True], 
         "Y": [Y[i-1]for i in positions if mutability[i-1] == True], 
         "Aln_best": [mutation_list_boa[i-1] for i in positions if mutability[i-1] == True],
         "Aln_seq": [mutation_list_boa_seq[i-1] for i in positions if mutability[i-1] == True], 
         "Aln_str": [mutation_list_boa_str[i-1] for i in positions if mutability[i-1] == True] }
    
#print(tabulate(table, headers = "keys",  tablefmt = "fancy_grid"))

# III.2. Creating output files with table and lists
with open(f'{working_directory}/table_{half_or_full_step}_to_{desired_charge}_{cs_lt}_{rsasa_gt}_{rmsf_gt}.txt', 'x') as output_file:
    output_file.write(tabulate(table, headers = "keys"))
    
#with open(f'{working_directory}/list_sizesuggest_{half_or_full_step}_to_{desired_charge}_{cs_lt}_{rsasa_gt}_{rmsf_gt}.txt', 'x') as output_file: 
    #for mutant in [i for i in mutation_list_bos if i != "-"]:
        #output_file.write("%s\n" % mutant)
        
#with open(f'{working_directory}/list_alignmentsuggest_to_{desired_charge}_{cs_lt}_{rsasa_gt}_{rmsf_gt}.txt', 'x') as output_file:
    #for mutant in [i for i in mutation_list_boa if i != "-"]:
        #output_file.write("%s\n" % mutant)
        
print("\nMutation suggestions based on size: " + str(len([mutation_list_bos[i-1] for i in positions if mutability[i-1] == True])))
print("Mutation suggestions based on multiple sequence alignment: " + str(len([mutation_list_boa[i-1] for i in positions if mutability[i-1] == True])))
print("Mutation suggestions based on sequence alignment with " + str(short_two) + ": " + str(len([mutation_list_boa_seq[i-1] for i in positions if mutability[i-1] == True and mutation_list_boa_seq[i-1] != "-"])))       
print("Mutation suggestions based on structure alignment with " + str(short_two) + ": " + str(len(alignment_mutations)))
print(str([i for i in alignment_mutations] if len(alignment_mutations) != 0 else ""))

print("\nOutput files generated: \n" + f'table_{half_or_full_step}_to_{desired_charge}_{cs_lt}_{rsasa_gt}_{rmsf_gt}.txt')
#print("\n" + f'list_sizesuggest_{half_or_full_step}_to_{desired_charge}_{cs_lt}_{rsasa_gt}_{rmsf_gt}.txt')
#print("\n" + f'list_alignmentsuggest_to_{desired_charge}_{cs_lt}_{rsasa_gt}_{rmsf_gt}.txt')


