

import os
print("CWD:", os.getcwd())
print("FILES:", os.listdir())
try:
    import pyximport
    pyximport.install()
    from draw_context_cython import DrawContext
except Exception as e:
    print("Falling back to Python DrawContext:", e)
    from draw_context import DrawContext

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import math
import random
# ##############################################################################
#specify input file names
# ##############################################################################
ncacx_filename = "ncacx_revisedtoy.txt"#"ncacx_revised1.txt" #pls update the ncacx data file name
ncocx_filename = "toynco.txt"#"ncocx_revised1.txt"#"ncocx_revised1.txt" #pls update the ncacx data file name
protein_seq = "ncacx_toy.txt"
run_num = 3 #define how many runs you want
final = 0#if final = 0, the output only contains those residues with both nca and nco definitely asgned
#if final = 1, the output contains those with not only nca and nco definitely asgned, but also nca asgned witn no nco match
#note: if you use output files as subsequent iterations, you should set final = 0.
diagnostics = 1 #will turn on neighbor and occupancy tracking
nstep = 40#120#40#30 #num of MCSA annealing steps
terminator = 40#120#30 #math.floor(0.5 * nstep) #determines from this step on, to the last step of MCSA,
#if a residue was not asgned a different asgn, then this residue reached its global minimum
#if rigor == 1: #only those asgn without entanglement will be counted towards consistently asgned to seed the next round
#if rigor == 2: only nmatch = 1 will be counted towards consistently asgned to seed the next round
#if rigor == 3: only those asgn nca and nco pair not have 15N overlap with others will be counted towards consistently asgned to seed the next rd
rigor = 0 #if rigor == 2:
 #those with overlap[npeak_nca] == 0 will not be penalized with their w1.
penalty = 1#10#2 #
#those with overlap[npeak_nca] = 0 will have their nngood penalized by penalty
#
delta_priority = 0.5 #if the sqrt((cs_calpha1-cs_calpha2)**2+(cs_cbeta1-cs_cbeta2)**2) < delta_priority,
#we treat them as identical in terms of secondary structure, and same as priority for local minimums.
#nfreq_nca = 5 #actual number of freq used for match finding
silencer = []#[26,38,72,75,91,152,215] #skip certain nca and all its matched nco in mc assign
# for any nca(and its matching nco asgns) to be skipped, add the nca's row index
#, for example, silencer = [1,2,3] will skip the first, second, and third entry in
#nca and all their associated matching nco signals in mc assign.
scale = 0.45#0.5 #scale down the w_i inc
#the following parameter you can adjust to control how rigorous the match finding is executed
nfreq_nca = 4 #actual number of freq used for match finding, you must find match for all specified number of sites, if they have valid inputs
ncac13lw_scalar = 1.0 # to amplify or shrink all carbon lws in direct dimensions of nca input
ncan15lw_scalar = 1.0 # to amplify or shrink all nitrogen lws in nca input
ncaclw_scalar = 1.0 #to amplify or shrink all carbon lws in CO dimension in nca input
disparity_nca1 = 1.0#0.33 #this is the ratio to reduce the delat_cs for the first indirect dimention to judge if a number of peaks in ncacx or ncocx should be counted towards the same residue, based on given uncertainty.
disparity_nca2 = 1.0#0.33 #this is the ratio to reduce the delat_cs for the second indirect dimention to judge if a number of peaks in ncacx or ncocx should be counted towards the same residue, based on given uncertainty.
ncoc13lw_scalar = 1.0 # to amplify or shrink all carbon lws in direct dimensions of nco input
ncon15lw_scalar = 1.0 # to amplify or shrink all nitrogen lws in nco input
ncoclw_scalar = 1.0 #to amplify or shrink all carbon lws in CO dimension in nco input
disparity_nco1 = 0.33 #this is the ratio to reduce the delat_cs for the first indirect dimention to judge if a number of peaks in ncacx or ncocx should be counted towards the same residue, based on given uncertainty.
disparity_nco2 = 0.33 #this is the ratio to reduce the delat_cs for the second indirect dimention to judge if a number of peaks in ncacx or ncocx should be counted towards the same residue, based on given uncertainty.

# ##############################################################################
# ###############################################################################
#specify relevant parameters for Monte Carlo Simulated Anealing
# ##############################################################################
conf_res='AGVITSP'#list of residues we can be sure of their residue type asgn
w1i = 0 # w1 initial value
w1f = 20#30#10#20#10 # w1 final value
w2i = 0 # w2 initial value
w2f = 50#60#20#50#20 # w2 final value
w3i = 0 # w3 initial value
w3f = 5#9#3#5#3 # w3 final value
w4i = 0 # w4 initial value
w4f = 10#3#1#10#1 # w4 final value

#number of successful Monte Carlo steps before go to next annealing step
nattempt =  5000#5000000
probzero_i = 0.4
# ##############################################################################
#specify output file names
# ##############################################################################
match_filename = "NCO_MatchSummary.txt"
matchdetail_filename = "NCO_MatchDetail.txt"
priority_filename = "priority.txt"
overlap_filename = "overlap.txt" #record the number of nca asgn that is similar to itself to form local minimums.
consistency_filename = "Consistency_summary.txt" #store the statistical consistency analysis of asg to each residue position by multiple runs specified by run_num
consistencynca_filename = "Consistencynca_summary.txt" #store the statistical consistency analysis of each asgn by multiple runs specified by run_num
neighbor_filename = "neighbor.txt"# track the nca asgn asgned to the neighboring positions in the sequence.
instigator_filename = "instigator.txt"# to measure the mobility of each nca asgn and the zero-th element is the total system mobility
occupancysum_filename = "occupancy_sum.txt"#record which nca asgn were allocated to each residue positioni in the sequence
occupancystep_filename = "occupancy_step.txt"#record at each MCSA step which nca asgn were allocated to each residue positioni in the sequence
knmatch_filename ="knmatch_rd.txt"#store the nmatch for current input
runrecord_filename = "runrecord.txt"#stores mc progress info.

ncacx4nextrd_filename ="ncacx4nextrd.txt"#store the ncacx for next round, with consistent asgn fixed
ncocx4nextrd_filename ="ncocx4nextrd.txt"#store the ncocx for next round, with consistent asgn fixed

ncacx4bk_basename ="ncacx4bk"#store the full ncacx asgn for each residue for each run
ncocx4bk_basename ="ncocx4bk"#store the full ncocx asgn for each residue for each run

numberrecord_filename ="number_record.txt"#4 x nstep matrix that records num of good, bad, edge and total usage along MCSA

detector_filename ="detector.txt"
runsummary_filename ="runsummary.txt"#records the tot_ngood, tot_nbad, tot_nedge, and tot_nused for each run. The first column is run number
###############################################################################
asgn_basename = "ncacx_assignmentNoLabel" #base_filename storing assignment detail
asgn_filename = []
ncacx4bk_filename = []
ncocx4bk_filename = []
for i in range(run_num):
    asgn_filename.append(asgn_basename + str(i) +'.txt')
    ncacx4bk_filename.append(ncacx4bk_basename + str(i) + '.txt')
    ncocx4bk_filename.append(ncocx4bk_basename + str(i) + '.txt')


#define some functions frequently used
###############################################################################
#function to extract only the letter part of a residue type asgn, not suitable
#if you want to extract a single letter from an ambiguous residue type asgm
def extract_letter(rtyp):
    #import re
    kresname=''
    for i in rtyp:
        if i.isalpha():
            kresname = "".join([kresname, i])
    return kresname
#end of extract letter function definition

#function to extract only the number part of a residue type asgn
def extract_number(rtyp):
    #import re
    kresnum=''
    for i in rtyp:
        if i.isdigit():
            kresnum = "".join([kresnum, i])
    return kresnum
#end of extract number function

#end of frequent used function definition
###############################################################################
def rtyp_compare(rtyp1, rtyp2):
    temp_len1 = len(extract_letter(rtyp1))
    temp_len2 = len(extract_letter(rtyp2))
    flag = 0
    for i1 in range(temp_len1):
        temp_rtyp1 = rtyp1[i1]
        for i2 in range(temp_len2):
            temp_rtyp2 = rtyp2[i2]
            if temp_rtyp1 == temp_rtyp2:
                flag += 1
    return flag

#Read in ncacx
###############################################################################
print('Read ncacx input from ',ncacx_filename,'.')
with open(ncacx_filename, "r") as file1:
    count = 0
    print(file1)
    for line in file1:
        count += 1
        #Read in the first row, how many residue type asgn in ncacx and
        #how many fre columns in each residue type asgn
        if count == 1:
            line1=line.split( )
            npeak_nca = int(line1[0])
            #nfreq_nca = int(line1[1])#actual number of freq used for match finding
            nfreqmax_nca = int(line1[1])#maximum number of freq input present, the max is 7 (including amide nitrogen and C', this means up to C-e is allowed in input, sufficient to differentiate any residue types)
            if nfreq_nca > nfreqmax_nca:
                nfreq_nca = nfreqmax_nca
                print('Warning: you required too many frequencies per residue than actual input!')

            n15freq_nca = [1e6 for i in range(npeak_nca+1)]
            n15lw_nca = [1e6 for i in range(npeak_nca+1)]
            ngoodfreq_nca= [0 for i in range(npeak_nca+1)]
            cfreq_nca= [1e6 for i in range(npeak_nca+1)]
            clw_nca= [1e6 for i in range(npeak_nca+1)]
            rows = npeak_nca+1
            cols = nfreqmax_nca-2
            csfreq_nca = [[1e6 for i in range(cols)] for j in range(rows)]
            cslw_nca = [[1e6 for i in range(cols)] for j in range(rows)]
            deg_nca = [1e6 for i in range(npeak_nca+1)]
            rtyp_nca = ['' for i in range(npeak_nca+1)]
            used_nca = [0 for i in range(npeak_nca+1)]
            nca_asgn = [0 for i in range(npeak_nca+1)]#track which residue position the nca asgn is asnged to
        #creat corresponding variables for each residue type asgn
        if count >1:
            kline=line.split()
            n15freq_nca[count-1] = float(kline[0])
            n15lw_nca[count-1] = float(kline[nfreqmax_nca])*ncan15lw_scalar
            cfreq_nca[count-1] = float(kline[2])
            clw_nca[count-1] = float(kline[2+nfreqmax_nca])*ncaclw_scalar
            deg_nca[count-1] = int(kline[2*nfreqmax_nca])
            rtyp_nca[count-1] = kline[2*nfreqmax_nca+1].strip()
            #rtyp_nca[count-1] = rtyp_nca[count-1].strip()
            used_nca[count-1] = 0

            csfreq_nca[count-1][0] = float(kline[1])#ca freq

            cslw_nca[count-1][0] = float(kline[1+nfreqmax_nca])*ncac13lw_scalar#ca lw
            for i1 in range(1,nfreqmax_nca-2):
                csfreq_nca[count-1][i1] = float(kline[i1+2])#side chain freq
                cslw_nca[count-1][i1] = float(kline[i1+2+nfreqmax_nca])*ncac13lw_scalar #side chain lw
        #now set the ngoodfreq_nca

for i1 in range(1,npeak_nca+1):
    temp_csfreq = []
    temp_cslw = []
    temp_len = 0
    for i2 in range(0,nfreq_nca-2):
        if csfreq_nca[i1][i2] < 200.0:
            ngoodfreq_nca[i1]=ngoodfreq_nca[i1]+1
            #the following is to deal the case where 1e6 was entered for a site before a valid entry
            temp_csfreq.append(csfreq_nca[i1][i2])
            temp_cslw.append(cslw_nca[i1][i2])
            temp_len = len(temp_csfreq)
    for i2 in range(0,temp_len):
        csfreq_nca[i1][i2] = temp_csfreq[i2]
        cslw_nca[i1][i2] = temp_cslw[i2]
    if temp_len < cols:
        for i2 in range(temp_len, cols):
            csfreq_nca[i1][i2] = 1e6
            cslw_nca[i1][i2] = 0.001


print('ncacx completes.')
print('\n')
#end of read in ncacx
###############################################################################

###############################################################################
#read in protein sequence
###############################################################################
print('Read in the protein sequence from ',protein_seq,'.')
#import re
protein=[]
protein_aa=''
with open(protein_seq, "r") as file1:
    count = 0
    lines=file1.readlines()


    for line in lines:
        for letter in line:
            if letter.isalpha():
                protein.append(letter)
                protein_aa = protein_aa + letter
    protein.insert(0,'')
    nres=len(protein)-1
    print('The number of residue in protein is ',nres)
    print(protein)
    nposs=[0 for i in range(nres+2)]
    poss=[[0] for i in range(nres+2)]



    nmatch=[0 for i in range(npeak_nca+1)]
    possmatch=[[0 for i in range(npeak_nca+1)] for j in range(1)]
    nposs[0] = 0
    poss[0] = 0
    nca_fix = [0 for i in range(npeak_nca+1)]
    for i2 in range(1, npeak_nca+1):
        #regex='\d+'#search pattern if there are multiple digits
        #match=re.findall(regex,rtyp_nca[i2])#check if there is sequential number asgn
        kresname=extract_letter(rtyp_nca[i2])
        kresnum=extract_number(rtyp_nca[i2])
        #if match:
            #print((match[0]))
            #kmatch = int(match[0])
        if len(kresnum) > 0:
            kmatch = int(kresnum)
            nposs[kmatch] = -1
            poss[kmatch].append(i2)
            poss[kmatch][1]=i2
            nca_fix[i2] = 1
            nca_asgn[i2] = kmatch

    for i1 in range(0, nres+1):
        for i2 in range(1, npeak_nca+1):
            #regex='\d+'#search pattern if there are multiple digits
            #match=re.findall(regex,rtyp_nca[i2])#check if there is sequential number asgn
            kresname=extract_letter(rtyp_nca[i2])
            kresnum=extract_number(rtyp_nca[i2])

            if len(kresnum) == 0:
                if nposs[i1] > -1:
                    cyc_len = len(kresname)
                    for i3 in range(0,cyc_len):
                        if protein[i1] == kresname[i3]:#in case we have ambiguous asgn, where multiple letters are present
                        #match = krtyp

                            poss[i1].append(i2)
                            nposs[i1]=nposs[i1]+1


print('Protein sequence input completes.')
print('\n')
#for i1 in range(1, nres+1):since we start the previous step from 0-th residue, all
#definite nca assignment would have been placed correctly without overflow the poss
    #if nposs[i1] == -1:
        #del poss[i1][2:] #remove excess points when the res is definitely asgned

###############################################################################
#end of read in protein sequence
###############################################################################
###############################################################################
#Read in ncocx
###############################################################################
print('Read in ncocx input from ',ncocx_filename,'.')
with open(ncocx_filename, "r") as file1:
    count = 0
    print(file1)
    for line in file1:
        count += 1
        #Read in the first row, how many residue type asgn in ncacx and
        #how many fre columns in each residue type asgn
        if count == 1:
            line1=line.split( )
            nsig_nco = 0
            nfreq_nco = int(line1[1])
            n15freq_nco = [1e6]
            n15lw_nco = [1e6]
            cfreq_nco= [1e6]
            clw_nco = [1e6]
            csfreq_nco = [1e6]
            cslw_nco = [1e6]
            deg_nco = [1e6]
            rtyp_nco=[' ']
            used_nco= [0]
            co_conf= [0]#track if a ncocx signal is matched to ncacx with confident CO asgn
            nco_row = [0]
        #creat corresponding variables for each residue type asgn
        if count >1:
            kline=line.split()
            n15freq_nco.append(float(kline[0]))
            n15lw_nco.append(float(kline[nfreq_nco]))
            cfreq_nco.append(float(kline[2]))
            clw_nco.append(float(kline[2+nfreq_nco]))
            deg_nco.append(int(kline[2*nfreq_nco]))
            used_nco.append(0)
            co_conf.append(0)
            rtyp_nco.append(kline[2*nfreq_nco+1].strip())
            #nco_row.append(count-1)#used to track the original row number of the nco signal
            #create the resonance corresponding to ca
            csfreq_nco.append(float(kline[1]))#ca freq
            cslw_nco.append(float(kline[1+nfreq_nco]))#ca lw
            nsig_nco = nsig_nco+1
            nco_row.append(count-1)#track its original row number
            #create resonances corresponding to each valid side-chain input
            for i1 in range(3,nfreq_nco):
                if float(kline[i1]) < 200:#if a side-chain resonance input is valid
                    nsig_nco = nsig_nco+1
                    #u need to generate corresponding N15 and CO parameters
                    n15freq_nco.append(float(kline[0]))
                    n15lw_nco.append(float(kline[nfreq_nco]))
                    cfreq_nco.append(float(kline[2]))
                    clw_nco.append(float(kline[2+nfreq_nco]))
                    #as well as its deg and usage parameters
                    deg_nco.append(int(kline[2*nfreq_nco]))
                    rtyp_nco.append(kline[2*nfreq_nco+1].strip())
                    used_nco.append(0)
                    co_conf.append(0)
                    csfreq_nco.append(float(kline[i1]))#ca freq
                    cslw_nco.append(float(kline[i1+nfreq_nco]))#ca lw
                    nco_row.append(count-1)#track its original row number
print('ncocx signal table input completes.')
print('\n')

#end of read in ncocx

#from draw_context import DrawContext

# … right here, after reading nres & npeak_nca …
# initialize current assignments per residue
current_nca = [0] * (nres + 1)
for nca_idx in range(1, npeak_nca + 1):
    res_idx = nca_asgn[nca_idx]
    if res_idx:
        current_nca[res_idx] = nca_idx

# start with no NCO assignments
current_nco = [0] * (nres + 1)


###############################################################################
###############################################################################
print('Match ncocx signal table to ncacx residue type assignments')
import time
start_time=time.time()
nmatch=[0 for i in range(npeak_nca+1)]
possmatch=[[0] for i in range(npeak_nca+1)]
#fix is to tracks if any of the signals used to match a ncacx residue type asgn is overused by any other match
#if none of the signals used for a ncacx asgn match is used for anything else,
#or if the used_nco for each of the signals used for a ncacx asgn match
#is less than its deg_nco,fix[nca_asgn index][nmatch]=1
#if the nco is input as sequentialled asgned, nmatch[nca_asgn]=-1, and fix[nca_asgn index][nmatch]=-1
#otherwise, fix[nca_asgn index][nmatch]=0
fix=[[0] for i in range(npeak_nca+1)]
#bound[i] equals one only if there is only a unique nco match to the i-th nca asgn
#the i-th nca asgn has to be (1).definitely asgned, or (2) residue type belonging to
#the confident assigned types conf_res, or (3) the sured asgned with CO freq, which
# is used to find the nco match. In summary, bound equals to one only for those absolute
#confident nca residue type asgn which found a unique nco match that is also
#absolutely confident.
#input for bound index is the nca asgn index, not residue in protein
bound=[0 for i in range(npeak_nca+1)]#bound[nca asgn]=1 if a nca has unique nco asgn and the used_nco of matched
#nco signals for this nca asgn is less than deg_nco of the signal.
#overuse is a list with the first two indexes coincides with possmatch to store the nco signal index that is overused in the match
overuse=[[0] for i in range(npeak_nca+1)]
def_nca=0
#skip_mc array, those residues with asigned definitely by nca and nco both, equal to 1.
skip_mc=[0 for i in range(nres+2)]#equal to 1 for res with definite nca and nco asgn


for i1 in range(1,npeak_nca+1):
    #print('current nca row is ',i1)
    ca_nmatch=0
    ca_possmatch=[0]
    if ngoodfreq_nca[i1]-1 >0:
        cb_nmatch=0
        cb_possmatch=[0]
        if ngoodfreq_nca[i1]-2 >0:
            cg_nmatch=0
            cg_possmatch=[0]
            if ngoodfreq_nca[i1]-3 >0:
                cd_nmatch=0
                cd_possmatch=[0]
                if ngoodfreq_nca[i1]-4 >0:
                    ce_nmatch=0
                    ce_possmatch=[0]


#########first we deal with ncacx residue type asgn with definite asgn
    if nca_fix[i1] == 1:#if current ncacx is a definite asgn

        flag=0

        krtyp_nca = extract_letter(rtyp_nca[i1])
        ctx.kresn_nca = extract_number(rtyp_nca[i1])
        for i2 in range(1,nsig_nco+1):
            #if used_nco[i2] < deg_nco[i2]:
                #if bool(re.search(r'\d+', rtyp_nco[i2])):#if the nco residue type asn contains number
            krtyp_nco = extract_letter(rtyp_nco[i2])
            kresn_nco = extract_number(rtyp_nco[i2])

            if krtyp_nca == krtyp_nco:
                if kresn_nca == kresn_nco:
                #make sure both residue typ and index are the same
                    #co_conf[i2]+=1
                    used_nco[i2]+=1

                    if flag == 0:
                        nmatch[i1]=1
                        bound[i1]=1
                        fix[i1].append([1])
                        temp_possmatch=[]
                        temp_possmatch.append(i2)
                        flag += 1
                        def_nca += 1
                        #co_conf[i2]+=1
                    elif flag == 1:
                        #print('i2 is ',i2)
                        #print('nco residue type is ',rtyp_nco[i2])
                        temp_possmatch.append(i2)
                        #co_conf[i2]+=1
        #print('i1 is ',i1)
        #print('nca residue type is ',rtyp_nca[i1])
        #print('i2 is ',i2)
        #print('nco residue type is ',rtyp_nco[i2])
        if flag == 1:
            match_len=len(temp_possmatch)
            len_diff=ngoodfreq_nca[i1] - match_len
            if len_diff > 0:#in case our definite nco asgn has less specified cs freq input than the matching definite nca asgn
                for k in range(1,len_diff+1):
                    temp_possmatch.append(0)#need to append 0, so it has its specified deg[0]=1e6
        if flag == 1:
            possmatch[i1].append(temp_possmatch)
            overuse[i1].append([[0]])
            bound[i1] = 1
#you need to re-order the peak in possmatch to ensure the match for ca is listed as the first element.
for i1 in range(1,npeak_nca+1):
    if nca_fix[i1] == 1:#if current ncacx is a definite asgn
        ksite_nca = csfreq_nca[i1][0]#select nca's ca freq
        klw_nca = cslw_nca[i1][0]
        if nmatch[i1] > 0:
            klen = len(possmatch[i1][1])
            for i2 in range(klen):
                site = possmatch[i1][1][i2]
                ksite_nco = csfreq_nco[site]
                klw_nco =cslw_nco[site]
                diff_freq = (ksite_nco - ksite_nca)**2
                delta_lw = klw_nca**2 + klw_nco **2
                if diff_freq < delta_lw :
                    ksite = site

            ksite_index = possmatch[i1][1].index(ksite)
            possmatch[i1][1].insert(0,possmatch[i1][1].pop(ksite_index))

for i2 in range(1,nsig_nco+1):
    #deg_nco[i2] = deg_nco[i2] - used_nco[i2]
    #used_nco[i2] = 0 #reset used_nco for this entry, so it can be used next to exclude those definite nca asgn with no definite nco pairing but was able to find match.
    if deg_nco[i2] - used_nco[i2] <= 0:
        co_conf[i2] = 1 #exclude signals marked by prior iterations for simultaneous def nca and nco asgn.

for i1 in range(1,npeak_nca+1):
    #print('current nca row is ',i1)
    ca_nmatch=0
    ca_possmatch=[0]
    if ngoodfreq_nca[i1]-1 >0:
        cb_nmatch=0
        cb_possmatch=[0]
        if ngoodfreq_nca[i1]-2 >0:
            cg_nmatch=0
            cg_possmatch=[0]
            if ngoodfreq_nca[i1]-3 >0:
                cd_nmatch=0
                cd_possmatch=[0]
                if ngoodfreq_nca[i1]-4 >0:
                    ce_nmatch=0
                    ce_possmatch=[0]

    if nca_fix[i1] == 1 and bound[i1] != 1:#if current ncacx is a definite asgn but previous step didnt find its paired definitely asgned nco

        for i2 in range(1,nsig_nco+1):
            if co_conf[i2] == 0:
                if cfreq_nca[i1] != 1e6 :
                    diff_c=cfreq_nca[i1]-cfreq_nco[i2]  # deviation between the C freq of a residue type assignment in ncacx and the C freq of a peak in raw_ncocx.
                    diff_c=diff_c*diff_c
                    delta_c= clw_nca[i1]**2+clw_nco[i2]**2 #FWHM linewidth tolerance check
                    if diff_c <=delta_c or cfreq_nca[i1] == 1e6 : #if this deviation is less than the linewidth, or the C freq of ncacx was not asgned
                         diff_cs=csfreq_nca[i1][0]-csfreq_nco[i2]
                         diff_cs=diff_cs*diff_cs
                         delta_cs=cslw_nca[i1][0]**2+ cslw_nco [i2]**2
                         delta_a=delta_cs-diff_cs
                         if delta_a >= 0:
                             ca_nmatch=ca_nmatch+1
                             ca_possmatch.append(i2)

                         if ngoodfreq_nca[i1]-1 >0:
                             diff_cs=csfreq_nca[i1][1]-csfreq_nco[i2]
                             diff_cs=diff_cs*diff_cs
                             delta_cs=cslw_nca[i1][1]**2+ cslw_nco [i2]**2
                             delta_b=delta_cs-diff_cs
                             if delta_b >= 0:
                                 cb_nmatch=cb_nmatch+1
                                 cb_possmatch.append(i2)

                             if ngoodfreq_nca[i1]-2 >0:
                                 diff_cs=csfreq_nca[i1][2]-csfreq_nco[i2]
                                 diff_cs=diff_cs*diff_cs
                                 delta_cs=cslw_nca[i1][2]**2+ cslw_nco [i2]**2
                                 delta_g=delta_cs-diff_cs
                                 if delta_g >= 0:
                                     cg_nmatch=cg_nmatch+1
                                     cg_possmatch.append(i2)

                                 if ngoodfreq_nca[i1]-3 >0:
                                     diff_cs=csfreq_nca[i1][3]-csfreq_nco[i2]
                                     diff_cs=diff_cs*diff_cs
                                     delta_cs=cslw_nca[i1][3]**2+ cslw_nco [i2]**2
                                     delta_d=delta_cs-diff_cs
                                     if delta_d >= 0:
                                         cd_nmatch=cd_nmatch+1
                                         cd_possmatch.append(i2)

                                     if ngoodfreq_nca[i1]-4 >0:
                                         diff_cs=csfreq_nca[i1][4]-csfreq_nco[i2]
                                         diff_cs=diff_cs*diff_cs
                                         delta_cs=cslw_nca[i1][4]**2+ cslw_nco [i2]**2
                                         delta_e=delta_cs-diff_cs
                                         if delta_e >= 0:
                                             ce_nmatch=ce_nmatch+1
                                             ce_possmatch.append(i2)



    ##################You need to compare each sig in nco with each site in csfreq_nca
        if ngoodfreq_nca[i1] == 1:
            if ca_nmatch>0:
                nmatch[i1] = ca_nmatch
                flag=1

                for k1 in range(1,ca_nmatch+1):
                    #nmatch[i1]= nmatch[i1]+1 #ue dont need to increment nmatch since only one carbon site to be matched
                    possmatch[i1].append([ca_possmatch[k1]])
                    fix[i1].append(0)
                    overuse[i1].append([[0]])
                    kca_nco=ca_possmatch[k1]
                    #co_conf[kca_nco]+=1
                    used_nco[kca_nco]+=1

        if ngoodfreq_nca[i1] == 2:
            if ca_nmatch*cb_nmatch>0:
                for k1 in range(1,ca_nmatch+1):
                    for k2 in range(1, cb_nmatch+1):
                        #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                        diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                        diff_n1=diff_n1**2
                        diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                        diff_nlw1=diff_nlw1*disparity_nco1**2

                        if diff_n1<=diff_nlw1:

                                nmatch[i1]= nmatch[i1]+1
                                possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2]])
                                fix[i1].append(0)
                                overuse[i1].append([[0]])
                                kca_nco=ca_possmatch[k1]
                                #co_conf[kca_nco]+=1
                                used_nco[kca_nco]+=1
                                kcb_nco=cb_possmatch[k2]
                                #co_conf[kcb_nco]+=1
                                used_nco[kcb_nco]+=1
                                flag=1

        if ngoodfreq_nca[i1] == 3:
            if ca_nmatch*cb_nmatch*cg_nmatch>0:
                for k1 in range(1,ca_nmatch+1):
                    for k2 in range(1, cb_nmatch+1):
                        #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                        diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                        diff_n1=diff_n1**2
                        diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                        diff_nlw1=diff_nlw1*disparity_nco1**2
                        if diff_n1<=diff_nlw1:
                           for k3 in range(1, cg_nmatch+1):
                                diff_n2=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cg_possmatch[k3]]
                                diff_n2=diff_n2**2
                                diff_nlw2=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cg_possmatch[k3]]**2
                                diff_nlw2=diff_nlw2*disparity_nco1**2
                                #their 15N freq must also align
                                if diff_n2<=diff_nlw2:
                                        nmatch[i1]= nmatch[i1]+1
                                        possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2],cg_possmatch[k3]])
                                        fix[i1].append(0)
                                        overuse[i1].append([[0]])
                                        kca_nco=ca_possmatch[k1]
                                        #co_conf[kca_nco]+=1
                                        used_nco[kca_nco]+=1
                                        kcb_nco=cb_possmatch[k2]
                                        #co_conf[kcb_nco]+=1
                                        used_nco[kcb_nco]+=1
                                        kcg_nco=cg_possmatch[k3]
                                        #co_conf[kcg_nco]+=1
                                        used_nco[kcg_nco]+=1
                                        flag=1

        if ngoodfreq_nca[i1] == 4:
            if ca_nmatch*cb_nmatch*cg_nmatch*cd_nmatch>0:
                for k1 in range(1,ca_nmatch+1):
                    for k2 in range(1, cb_nmatch+1):
                        #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                        diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                        diff_n1=diff_n1**2
                        diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                        diff_nlw1=diff_nlw1*disparity_nco1**2
                        if diff_n1<=diff_nlw1:
                            for k3 in range(1, cg_nmatch+1):
                                diff_n2=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cg_possmatch[k3]]
                                diff_n2=diff_n2**2
                                diff_nlw2=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cg_possmatch[k3]]**2
                                diff_nlw2=diff_nlw2*disparity_nco1**2
                                #their 15N freq must also align
                                if diff_n2<=diff_nlw2:
                                    for k4 in range(1, cd_nmatch+1):
                                        diff_n3=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cd_possmatch[k4]]
                                        diff_n3=diff_n3**2
                                        diff_nlw3=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cd_possmatch[k4]]**2
                                        diff_nlw3=diff_nlw3*disparity_nco1**2
                                        if diff_n3<=diff_nlw3:
                                            nmatch[i1]= nmatch[i1]+1
                                            possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2],cg_possmatch[k3],cd_possmatch[k4]])
                                            fix[i1].append(0)
                                            overuse[i1].append([[0]])
                                            kca_nco=ca_possmatch[k1]
                                            #co_conf[kca_nco]+=1
                                            used_nco[kca_nco]+=1
                                            kcb_nco=cb_possmatch[k2]
                                            #co_conf[kcb_nco]+=1
                                            used_nco[kcb_nco]+=1
                                            kcg_nco=cg_possmatch[k3]
                                            #co_conf[kcg_nco]+=1
                                            used_nco[kcg_nco]+=1
                                            kcd_nco=cd_possmatch[k4]
                                            #co_conf[kcd_nco]+=1
                                            used_nco[kcd_nco]+=1
                                            flag=1

        if ngoodfreq_nca[i1] == 5:
            if ca_nmatch*cb_nmatch*cg_nmatch*cd_nmatch*ce_nmatch>0:
                for k1 in range(1,ca_nmatch+1):
                    for k2 in range(1, cb_nmatch+1):
                        #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                        diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                        diff_n1=diff_n1**2
                        diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                        diff_nlw1=diff_nlw1*disparity_nco1**2
                        if diff_n1<=diff_nlw1:
                            for k3 in range(1, cg_nmatch+1):
                                diff_n2=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cg_possmatch[k3]]
                                diff_n2=diff_n2**2
                                diff_nlw2=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cg_possmatch[k3]]**2
                                diff_nlw2=diff_nlw2*disparity_nco1**2
                                #their 15N freq must also align
                                if diff_n2<=diff_nlw2:
                                    for k4 in range(1, cd_nmatch+1):
                                        diff_n3=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cd_possmatch[k4]]
                                        diff_n3=diff_n3**2
                                        diff_nlw3=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cd_possmatch[k4]]**2
                                        diff_nlw3=diff_nlw3*disparity_nco1**2
                                        if diff_n3<=diff_nlw3:
                                            for k5 in range(1, ce_nmatch+1):
                                                diff_n4=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[ce_possmatch[k5]]
                                                diff_n4=diff_n4**2
                                                diff_nlw4=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[ce_possmatch[k5]]**2
                                                diff_nlw4=diff_nlw4*disparity_nco1**2
                                                if diff_n4<=diff_nlw4:
                                                    nmatch[i1]= nmatch[i1]+1
                                                    possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2],cg_possmatch[k3],cd_possmatch[k4],ce_possmatch[k5]])
                                                    fix[i1].append(0)
                                                    overuse[i1].append([[0]])
                                                    kca_nco=ca_possmatch[k1]
                                                    #co_conf[kca_nco]+=1
                                                    used_nco[kca_nco]+=1
                                                    kcb_nco=cb_possmatch[k2]
                                                    #co_conf[kcb_nco]+=1
                                                    used_nco[kcb_nco]+=1
                                                    kcg_nco=cg_possmatch[k3]
                                                    #co_conf[kcg_nco]+=1
                                                    used_nco[kcg_nco]+=1
                                                    kcd_nco=cd_possmatch[k4]
                                                    #co_conf[kcd_nco]+=1
                                                    used_nco[kcd_nco]+=1
                                                    kce_nco=ce_possmatch[k5]
                                                    #co_conf[kce_nco]+=1
                                                    used_nco[kce_nco]+=1
                                                    flag = 1
#for i2 in range(1,nsig_nco+1):
    #deg_nco[i2] = deg_nco[i2] - used_nco[i2]
    #if deg_nco[i2] <= 0:
        #co_conf[i2] = 1 #exclude signals marked by prior iterations for simultaneous def nca and nco asgn.

#########second we deal with ncacx residue type asgn with confident co freq asgn
#for i1 in range(1,npeak_nca+1):

    #print('current nca row is ',i1)
    elif nca_fix[i1] == 0: #exclude the nca with definite asgn now
        #print('current nca row is ',i1)
        ca_nmatch=0
        ca_possmatch=[0]
        if ngoodfreq_nca[i1]-1 >0:
            cb_nmatch=0
            cb_possmatch=[0]
            if ngoodfreq_nca[i1]-2 >0:
                cg_nmatch=0
                cg_possmatch=[0]
                if ngoodfreq_nca[i1]-3 >0:
                    cd_nmatch=0
                    cd_possmatch=[0]
                    if ngoodfreq_nca[i1]-4 >0:
                        ce_nmatch=0
                        ce_possmatch=[0]
        if cfreq_nca[i1] < 200.0:
            for i2 in range(1,nsig_nco+1):
                #in this free nco match version, if nco is specified for a ncacx residue
                #type assignment, we will use it to scrutinize matched nco asgn.
                #If it is not specified in a ncacx
                #residue type assignment, we assume the co assignment in nca
                #is not reliable, so we cant require the co of the residue type asgn in
                #nca to match with the co freq in nco. Instead, we require the co freq
                #of each peak in nco to align, if the group of peaks are considered to
                #arise from the same residue in ncocx, just like their nitrogen freqs
                #need to align.
                if co_conf[i2] == 0: #exclude the nco signals with definite asgn
                    diff_c=cfreq_nca[i1]-cfreq_nco[i2]  # deviation between the C freq of a residue type assignment in ncacx and the C freq of a peak in raw_ncocx.
                    diff_c=diff_c*diff_c
                    delta_c= clw_nca[i1]**2+clw_nco[i2]**2 #FWHM linewidth tolerance check
                    if diff_c <=delta_c : #if this deviation is less than the linewidth
                         diff_cs=csfreq_nca[i1][0]-csfreq_nco[i2]
                         diff_cs=diff_cs*diff_cs
                         delta_cs=cslw_nca[i1][0]**2+ cslw_nco [i2]**2
                         delta_a=delta_cs-diff_cs
                         if delta_a >= 0:
                             ca_nmatch=ca_nmatch+1
                             ca_possmatch.append(i2)

                         if ngoodfreq_nca[i1]-1 >0:
                             diff_cs=csfreq_nca[i1][1]-csfreq_nco[i2]
                             diff_cs=diff_cs*diff_cs
                             delta_cs=cslw_nca[i1][1]**2+ cslw_nco [i2]**2
                             delta_b=delta_cs-diff_cs
                             if delta_b >= 0:
                                 cb_nmatch=cb_nmatch+1
                                 cb_possmatch.append(i2)

                             if ngoodfreq_nca[i1]-2 >0:
                                 diff_cs=csfreq_nca[i1][2]-csfreq_nco[i2]
                                 diff_cs=diff_cs*diff_cs
                                 delta_cs=cslw_nca[i1][2]**2+ cslw_nco [i2]**2
                                 delta_g=delta_cs-diff_cs
                                 if delta_g >= 0:
                                     cg_nmatch=cg_nmatch+1
                                     cg_possmatch.append(i2)

                                 if ngoodfreq_nca[i1]-3 >0:
                                     diff_cs=csfreq_nca[i1][3]-csfreq_nco[i2]
                                     diff_cs=diff_cs*diff_cs
                                     delta_cs=cslw_nca[i1][3]**2+ cslw_nco [i2]**2
                                     delta_d=delta_cs-diff_cs
                                     if delta_d >= 0:
                                         cd_nmatch=cd_nmatch+1
                                         cd_possmatch.append(i2)

                                     if ngoodfreq_nca[i1]-4 >0:
                                         diff_cs=csfreq_nca[i1][4]-csfreq_nco[i2]
                                         diff_cs=diff_cs*diff_cs
                                         delta_cs=cslw_nca[i1][4]**2+ cslw_nco [i2]**2
                                         delta_e=delta_cs-diff_cs
                                         if delta_e >= 0:
                                             ce_nmatch=ce_nmatch+1
                                             ce_possmatch.append(i2)



        ##################You need to compare each sig in nco with each site in csfreq_nca
            if ngoodfreq_nca[i1] == 1:
                if ca_nmatch>0:
                    nmatch[i1] = ca_nmatch

                    for k1 in range(1,ca_nmatch+1):
                        #nmatch[i1]= nmatch[i1]+1 #ue dont need to increment nmatch since only one carbon site to be matched
                        possmatch[i1].append([ca_possmatch[k1]])
                        fix[i1].append(0)
                        overuse[i1].append([[0]])
                        kca_nco=ca_possmatch[k1]
                        #co_conf[kca_nco]+=1
                        used_nco[kca_nco]+=1

            if ngoodfreq_nca[i1] == 2:
                if ca_nmatch*cb_nmatch>0:
                    for k1 in range(1,ca_nmatch+1):
                        for k2 in range(1, cb_nmatch+1):
                            #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                            diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                            diff_n1=diff_n1**2
                            diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                            diff_nlw1=diff_nlw1*disparity_nco1**2

                            if diff_n1<=diff_nlw1:

                                    nmatch[i1]= nmatch[i1]+1
                                    possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2]])
                                    fix[i1].append(0)
                                    overuse[i1].append([[0]])
                                    kca_nco=ca_possmatch[k1]
                                    #co_conf[kca_nco]+=1
                                    used_nco[kca_nco]+=1
                                    kcb_nco=cb_possmatch[k2]
                                    #co_conf[kcb_nco]+=1
                                    used_nco[kcb_nco]+=1

            if ngoodfreq_nca[i1] == 3:
                if ca_nmatch*cb_nmatch*cg_nmatch>0:
                    for k1 in range(1,ca_nmatch+1):
                        for k2 in range(1, cb_nmatch+1):
                            #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                            diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                            diff_n1=diff_n1**2
                            diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                            diff_nlw1=diff_nlw1*disparity_nco1**2
                            if diff_n1<=diff_nlw1:
                               for k3 in range(1, cg_nmatch+1):
                                    diff_n2=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cg_possmatch[k3]]
                                    diff_n2=diff_n2**2
                                    diff_nlw2=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cg_possmatch[k3]]**2
                                    diff_nlw2=diff_nlw2*disparity_nco1**2
                                    #their 15N freq must also align
                                    if diff_n2<=diff_nlw2:
                                            nmatch[i1]= nmatch[i1]+1
                                            possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2],cg_possmatch[k3]])
                                            fix[i1].append(0)
                                            overuse[i1].append([[0]])
                                            kca_nco=ca_possmatch[k1]
                                            #co_conf[kca_nco]+=1
                                            used_nco[kca_nco]+=1
                                            kcb_nco=cb_possmatch[k2]
                                            #co_conf[kcb_nco]+=1
                                            used_nco[kcb_nco]+=1
                                            kcg_nco=cg_possmatch[k3]
                                            #co_conf[kcg_nco]+=1
                                            used_nco[kcg_nco]+=1

            if ngoodfreq_nca[i1] == 4:
                if ca_nmatch*cb_nmatch*cg_nmatch*cd_nmatch>0:
                    for k1 in range(1,ca_nmatch+1):
                        for k2 in range(1, cb_nmatch+1):
                            #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                            diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                            diff_n1=diff_n1**2
                            diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                            diff_nlw1=diff_nlw1*disparity_nco1**2
                            if diff_n1<=diff_nlw1:
                                for k3 in range(1, cg_nmatch+1):
                                    diff_n2=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cg_possmatch[k3]]
                                    diff_n2=diff_n2**2
                                    diff_nlw2=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cg_possmatch[k3]]**2
                                    diff_nlw2=diff_nlw2*disparity_nco1**2
                                    #their 15N freq must also align
                                    if diff_n2<=diff_nlw2:
                                        for k4 in range(1, cd_nmatch+1):
                                            diff_n3=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cd_possmatch[k4]]
                                            diff_n3=diff_n3**2
                                            diff_nlw3=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cd_possmatch[k4]]**2
                                            diff_nlw3=diff_nlw3*disparity_nco1**2
                                            if diff_n3<=diff_nlw3:
                                                nmatch[i1]= nmatch[i1]+1
                                                possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2],cg_possmatch[k3],cd_possmatch[k4]])
                                                fix[i1].append(0)
                                                overuse[i1].append([[0]])
                                                kca_nco=ca_possmatch[k1]
                                                #co_conf[kca_nco]+=1
                                                used_nco[kca_nco]+=1
                                                kcb_nco=cb_possmatch[k2]
                                                #co_conf[kcb_nco]+=1
                                                used_nco[kcb_nco]+=1
                                                kcg_nco=cg_possmatch[k3]
                                                #co_conf[kcg_nco]+=1
                                                used_nco[kcg_nco]+=1
                                                kcd_nco=cd_possmatch[k4]
                                                #co_conf[kcd_nco]+=1
                                                used_nco[kcd_nco]+=1

            if ngoodfreq_nca[i1] == 5:
                if ca_nmatch*cb_nmatch*cg_nmatch*cd_nmatch*ce_nmatch>0:
                    for k1 in range(1,ca_nmatch+1):
                        for k2 in range(1, cb_nmatch+1):
                            #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                            diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                            diff_n1=diff_n1**2
                            diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                            diff_nlw1=diff_nlw1*disparity_nco1**2
                            if diff_n1<=diff_nlw1:
                                for k3 in range(1, cg_nmatch+1):
                                    diff_n2=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cg_possmatch[k3]]
                                    diff_n2=diff_n2**2
                                    diff_nlw2=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cg_possmatch[k3]]**2
                                    diff_nlw2=diff_nlw2*disparity_nco1**2
                                    #their 15N freq must also align
                                    if diff_n2<=diff_nlw2:
                                        for k4 in range(1, cd_nmatch+1):
                                            diff_n3=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cd_possmatch[k4]]
                                            diff_n3=diff_n3**2
                                            diff_nlw3=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cd_possmatch[k4]]**2
                                            diff_nlw3=diff_nlw3*disparity_nco1**2
                                            if diff_n3<=diff_nlw3:
                                                for k5 in range(1, ce_nmatch+1):
                                                    diff_n4=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[ce_possmatch[k5]]
                                                    diff_n4=diff_n4**2
                                                    diff_nlw4=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[ce_possmatch[k5]]**2
                                                    diff_nlw4=diff_nlw4*disparity_nco1**2
                                                    if diff_n4<=diff_nlw4:
                                                        nmatch[i1]= nmatch[i1]+1
                                                        possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2],cg_possmatch[k3],cd_possmatch[k4],ce_possmatch[k5]])
                                                        fix[i1].append(0)
                                                        overuse[i1].append([[0]])
                                                        kca_nco=ca_possmatch[k1]
                                                        #co_conf[kca_nco]+=1
                                                        used_nco[kca_nco]+=1
                                                        kcb_nco=cb_possmatch[k2]
                                                        #co_conf[kcb_nco]+=1
                                                        used_nco[kcb_nco]+=1
                                                        kcg_nco=cg_possmatch[k3]
                                                        #co_conf[kcg_nco]+=1
                                                        used_nco[kcg_nco]+=1
                                                        kcd_nco=cd_possmatch[k4]
                                                        #co_conf[kcd_nco]+=1
                                                        used_nco[kcd_nco]+=1
                                                        kce_nco=ce_possmatch[k5]
                                                        #co_conf[kce_nco]+=1
                                                        used_nco[kce_nco]+=1

    #end of match for the second cases where CO freq in ncacx residue type is certain
    #now we deal with the third cases ncacx residue type asgn with uncertain CO asgn(no co freq in ncacx)
        #if i1 == 36:

#for i1 in range(1, npeak_nca+1):
    #print('current nca row is ',i1)
    #if nca_fix[i1]==0:
        #if nca_fix[i1]==0:

        elif cfreq_nca[i1] == 1e6:
            #first deal with residues with confident residue type asgn such as VGITSP, defined in conf_res
            krtyp_nca=extract_letter(rtyp_nca[i1])
            #extract only the letter part of residue type asgn
            if krtyp_nca in conf_res:
                krtyp_nca=extract_letter(rtyp_nca[i1])
                #extract only the letter part of residue type asgn
                #print('current res type is ', krtyp_nca)
                for i2 in range(1,nsig_nco+1):
                    if co_conf[i2] == 0: #exclude nco signals used for definite nco asgn.
                        diff_cs=csfreq_nca[i1][0]-csfreq_nco[i2]
                        diff_cs=diff_cs*diff_cs
                        delta_cs=cslw_nca[i1][0]**2+ cslw_nco [i2]**2
                        delta_a=delta_cs-diff_cs
                        if delta_a >= 0:
                            ca_nmatch=ca_nmatch+1
                            ca_possmatch.append(i2)

                        if ngoodfreq_nca[i1]-1 >0:
                            diff_cs=csfreq_nca[i1][1]-csfreq_nco[i2]
                            diff_cs=diff_cs*diff_cs
                            delta_cs=cslw_nca[i1][1]**2+ cslw_nco [i2]**2
                            delta_b=delta_cs-diff_cs
                            if delta_b >= 0:
                                cb_nmatch=cb_nmatch+1
                                cb_possmatch.append(i2)

                            if ngoodfreq_nca[i1]-2 >0:
                                diff_cs=csfreq_nca[i1][2]-csfreq_nco[i2]
                                diff_cs=diff_cs*diff_cs
                                delta_cs=cslw_nca[i1][2]**2+ cslw_nco [i2]**2
                                delta_g=delta_cs-diff_cs
                                if delta_g >= 0:
                                    cg_nmatch=cg_nmatch+1
                                    cg_possmatch.append(i2)

                                if ngoodfreq_nca[i1]-3 >0:
                                    diff_cs=csfreq_nca[i1][3]-csfreq_nco[i2]
                                    diff_cs=diff_cs*diff_cs
                                    delta_cs=cslw_nca[i1][3]**2+ cslw_nco [i2]**2
                                    delta_d=delta_cs-diff_cs
                                    if delta_d >= 0:
                                        cd_nmatch=cd_nmatch+1
                                        cd_possmatch.append(i2)

                                    if ngoodfreq_nca[i1]-4 >0:
                                        diff_cs=csfreq_nca[i1][4]-csfreq_nco[i2]
                                        diff_cs=diff_cs*diff_cs
                                        delta_cs=cslw_nca[i1][4]**2+ cslw_nco [i2]**2
                                        delta_e=delta_cs-diff_cs
                                        if delta_e >= 0:
                                            ce_nmatch=ce_nmatch+1
                                            ce_possmatch.append(i2)
                ##################You need to compare each sig in nco with each site in csfreq_nca
                if ngoodfreq_nca[i1] == 1:
                    if ca_nmatch>0:
                        nmatch[i1] = ca_nmatch

                        for k1 in range(1,ca_nmatch+1):
                            #nmatch[i1]= nmatch[i1]+1 #ue dont need to increment nmatch since only one carbon site to be matched
                            possmatch[i1].append([ca_possmatch[k1]])
                            fix[i1].append([0])
                            overuse[i1].append([[0]])
                            kca_nco=ca_possmatch[k1]
                            used_nco[kca_nco]+=1

                if ngoodfreq_nca[i1] == 2:
                    if ca_nmatch*cb_nmatch>0:
                        for k1 in range(1,ca_nmatch+1):
                            for k2 in range(1, cb_nmatch+1):
                                #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                                diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                                diff_n1=diff_n1**2
                                diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                                diff_nlw1=diff_nlw1*disparity_nco1**2

                                if diff_n1<=diff_nlw1:
                                    #their CO freq must also align
                                    diff_co1=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cb_possmatch[k2]]
                                    diff_co1=diff_co1**2
                                    diff_colw1=clw_nco[ca_possmatch[k1]]**2+clw_nco[cb_possmatch[k2]]**2
                                    diff_colw1=diff_colw1*disparity_nco2**2
                                    if diff_co1<=diff_colw1:
                                        nmatch[i1]= nmatch[i1]+1
                                        possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2]])
                                        fix[i1].append([0])
                                        overuse[i1].append([[0]])
                                        kca_nco=ca_possmatch[k1]
                                        used_nco[kca_nco]+=1
                                        kcb_nco=cb_possmatch[k2]
                                        used_nco[kcb_nco]+=1


                if ngoodfreq_nca[i1] == 3:
                    if ca_nmatch*cb_nmatch*cg_nmatch>0:
                        for k1 in range(1,ca_nmatch+1):
                            for k2 in range(1, cb_nmatch+1):
                                #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                                diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                                diff_n1=diff_n1**2
                                diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                                diff_nlw1=diff_nlw1*disparity_nco1**2
                                if diff_n1<=diff_nlw1:
                                    diff_co1=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cb_possmatch[k2]]
                                    diff_co1=diff_co1**2
                                    diff_colw1=clw_nco[ca_possmatch[k1]]**2+clw_nco[cb_possmatch[k2]]**2
                                    diff_colw1=diff_colw1*disparity_nco2**2
                                    if diff_co1<=diff_colw1:
                                        for k3 in range(1, cg_nmatch+1):
                                            diff_n2=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cg_possmatch[k3]]
                                            diff_n2=diff_n2**2
                                            diff_nlw2=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cg_possmatch[k3]]**2
                                            diff_nlw2=diff_nlw2*disparity_nco1**2
                                            #their CO freq must also align
                                            if diff_n2<=diff_nlw2:
                                                diff_co2=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cg_possmatch[k3]]
                                                diff_co2=diff_co2**2
                                                diff_colw2=clw_nco[ca_possmatch[k1]]**2+clw_nco[cg_possmatch[k3]]**2
                                                diff_colw2=diff_colw2*disparity_nco2**2
                                                if diff_co2<=diff_colw2:
                                                    nmatch[i1]= nmatch[i1]+1
                                                    possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2],cg_possmatch[k3]])
                                                    fix[i1].append([0])
                                                    overuse[i1].append([[0]])
                                                    kca_nco=ca_possmatch[k1]
                                                    used_nco[kca_nco]+=1
                                                    kcb_nco=cb_possmatch[k2]
                                                    used_nco[kcb_nco]+=1
                                                    kcg_nco=cg_possmatch[k3]
                                                    used_nco[kcg_nco]+=1


                if ngoodfreq_nca[i1] == 4:
                    if ca_nmatch*cb_nmatch*cg_nmatch*cd_nmatch>0:
                        for k1 in range(1,ca_nmatch+1):
                            for k2 in range(1, cb_nmatch+1):
                                #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                                diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                                diff_n1=diff_n1**2
                                diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                                diff_nlw1=diff_nlw1*disparity_nco1**2
                                if diff_n1<=diff_nlw1:
                                    diff_co1=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cb_possmatch[k2]]
                                    diff_co1=diff_co1**2
                                    diff_colw1=clw_nco[ca_possmatch[k1]]**2+clw_nco[cb_possmatch[k2]]**2
                                    diff_colw1=diff_colw1*disparity_nco2**2
                                    if diff_co1<=diff_colw1:
                                        for k3 in range(1, cg_nmatch+1):
                                            diff_n2=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cg_possmatch[k3]]
                                            diff_n2=diff_n2**2
                                            diff_nlw2=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cg_possmatch[k3]]**2
                                            diff_nlw2=diff_nlw2*disparity_nco1**2
                                            #their CO freq must also align
                                            if diff_n2<=diff_nlw2:
                                                diff_co2=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cg_possmatch[k3]]
                                                diff_co2=diff_co2**2
                                                diff_colw2=clw_nco[ca_possmatch[k1]]**2+clw_nco[cg_possmatch[k3]]**2
                                                diff_colw2=diff_colw2*disparity_nco2**2
                                                if diff_co2<=diff_colw2:
                                                    for k4 in range(1, cd_nmatch+1):
                                                        diff_n3=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cd_possmatch[k4]]
                                                        diff_n3=diff_n3**2
                                                        diff_nlw3=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cd_possmatch[k4]]**2
                                                        diff_nlw3=diff_nlw3*disparity_nco1**2
                                                        if diff_n3<=diff_nlw3:
                                                            diff_co3=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cd_possmatch[k4]]
                                                            diff_co3=diff_co3**2
                                                            diff_colw3=clw_nco[ca_possmatch[k1]]**2+clw_nco[cd_possmatch[k4]]**2
                                                            diff_colw3=diff_colw3*disparity_nco2**2
                                                            if diff_co3<=diff_colw3:
                                                                nmatch[i1]= nmatch[i1]+1
                                                                possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2],cg_possmatch[k3],cd_possmatch[k4]])
                                                                fix[i1].append([0])
                                                                overuse[i1].append([[0]])
                                                                kca_nco=ca_possmatch[k1]
                                                                used_nco[kca_nco]+=1
                                                                kcb_nco=cb_possmatch[k2]
                                                                used_nco[kcb_nco]+=1
                                                                kcg_nco=cg_possmatch[k3]
                                                                used_nco[kcg_nco]+=1
                                                                kcd_nco=cd_possmatch[k4]
                                                                used_nco[kcd_nco]+=1


                if ngoodfreq_nca[i1] == 5:
                    if ca_nmatch*cb_nmatch*cg_nmatch*cd_nmatch*ce_nmatch>0:
                        for k1 in range(1,ca_nmatch+1):
                            for k2 in range(1, cb_nmatch+1):
                                #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                                diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                                diff_n1=diff_n1**2
                                diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                                diff_nlw1=diff_nlw1*disparity_nco1**2
                                if diff_n1<=diff_nlw1:
                                    diff_co1=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cb_possmatch[k2]]
                                    diff_co1=diff_co1**2
                                    diff_colw1=clw_nco[ca_possmatch[k1]]**2+clw_nco[cb_possmatch[k2]]**2
                                    diff_colw1=diff_colw1*disparity_nco2**2
                                    if diff_co1<=diff_colw1:
                                        for k3 in range(1, cg_nmatch+1):
                                            diff_n2=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cg_possmatch[k3]]
                                            diff_n2=diff_n2**2
                                            diff_nlw2=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cg_possmatch[k3]]**2
                                            diff_nlw2=diff_nlw2*disparity_nco1**2
                                            #their CO freq must also align
                                            if diff_n2<=diff_nlw2:
                                                diff_co2=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cg_possmatch[k3]]
                                                diff_co2=diff_co2**2
                                                diff_colw2=clw_nco[ca_possmatch[k1]]**2+clw_nco[cg_possmatch[k3]]**2
                                                diff_colw2=diff_colw2*disparity_nco2**2
                                                if diff_co2<=diff_colw2:
                                                    for k4 in range(1, cd_nmatch+1):
                                                        diff_n3=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cd_possmatch[k4]]
                                                        diff_n3=diff_n3**2
                                                        diff_nlw3=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cd_possmatch[k4]]**2
                                                        diff_nlw3=diff_nlw3*disparity_nco1**2
                                                        if diff_n3<=diff_nlw3:
                                                            diff_co3=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cd_possmatch[k4]]
                                                            diff_co3=diff_co3**2
                                                            diff_colw3=clw_nco[ca_possmatch[k1]]**2+clw_nco[cd_possmatch[k4]]**2
                                                            diff_colw3=diff_colw3*disparity_nco2**2
                                                            if diff_co3<=diff_colw3:
                                                                for k5 in range(1, ce_nmatch+1):
                                                                    diff_n4=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[ce_possmatch[k5]]
                                                                    diff_n4=diff_n4**2
                                                                    diff_nlw4=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[ce_possmatch[k5]]**2
                                                                    diff_nlw4=diff_nlw4*disparity_nco1**2
                                                                    if diff_n4<=diff_nlw4:
                                                                        diff_co4=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[ce_possmatch[k5]]
                                                                        diff_co4=diff_co4**2
                                                                        diff_colw4=clw_nco[ca_possmatch[k1]]**2+clw_nco[ce_possmatch[k5]]**2
                                                                        diff_colw4=diff_colw1*disparity_nco2**2
                                                                        if diff_co3<=diff_colw3:
                                                                            nmatch[i1]= nmatch[i1]+1
                                                                            possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2],cg_possmatch[k3],cd_possmatch[k4],ce_possmatch[k5]])
                                                                            fix[i1].append([0])
                                                                            overuse[i1].append([[0]])
                                                                            kca_nco=ca_possmatch[k1]
                                                                            used_nco[kca_nco]+=1
                                                                            kcb_nco=cb_possmatch[k2]
                                                                            used_nco[kcb_nco]+=1
                                                                            kcg_nco=cg_possmatch[k3]
                                                                            used_nco[kcg_nco]+=1
                                                                            kcd_nco=cd_possmatch[k4]
                                                                            used_nco[kcd_nco]+=1
                                                                            kce_nco=ce_possmatch[k5]
                                                                            used_nco[kce_nco]+=1
            #end of third cases: finding match for those nca asgn with conf_res type but no CO freq.

            #now for the fourth cases, for those nca asgn with residue type not those confident type, or without CO asgn    #this means nco signals used for nca asgn with CO freq, or nca asgn without CO freq but residue type asgn falling in the conf_res list.
#for i1 in range(1, npeak_nca+1):

    #if nca_fix[i1]==0:
        #if nca_fix[i1]==0:
        #if cfreq_nca[i1] == 1e6:
            #krtyp_nca=extract_letter(rtyp_nca[i1])
            #extract only the letter part of residue type asgn
            #print('current res type is ', krtyp_nca)
            #if krtyp_nca in conf_res:
                #ok=krtyp_nca
                #print('match res type for nca row ',i1)
            else:
                #print('current nca row is ',i1,', current nca_fix is ', nca_fix[i1],', current restype is ',krtyp_nca)
                for i2 in range(1,nsig_nco+1):
                    if co_conf[i2] == 0: #exclude nco signals used for definite nco asgn.
                        #if used_nco[i2] <:#if the nco signal was not previously matched for ncacx residue type asgn with confident residue type asgn.

                        diff_cs=csfreq_nca[i1][0]-csfreq_nco[i2]
                        diff_cs=diff_cs*diff_cs
                        delta_cs=cslw_nca[i1][0]**2+ cslw_nco [i2]**2
                        delta_a=delta_cs-diff_cs
                        if delta_a >= 0:
                            ca_nmatch=ca_nmatch+1
                            ca_possmatch.append(i2)

                        if ngoodfreq_nca[i1]-1 >0:
                            diff_cs=csfreq_nca[i1][1]-csfreq_nco[i2]
                            diff_cs=diff_cs*diff_cs
                            delta_cs=cslw_nca[i1][1]**2+ cslw_nco [i2]**2
                            delta_b=delta_cs-diff_cs
                            if delta_b >= 0:
                                cb_nmatch=cb_nmatch+1
                                cb_possmatch.append(i2)

                            if ngoodfreq_nca[i1]-2 >0:
                                diff_cs=csfreq_nca[i1][2]-csfreq_nco[i2]
                                diff_cs=diff_cs*diff_cs
                                delta_cs=cslw_nca[i1][2]**2+ cslw_nco [i2]**2
                                delta_g=delta_cs-diff_cs
                                if delta_g >= 0:
                                    cg_nmatch=cg_nmatch+1
                                    cg_possmatch.append(i2)

                                if ngoodfreq_nca[i1]-3 >0:
                                    diff_cs=csfreq_nca[i1][3]-csfreq_nco[i2]
                                    diff_cs=diff_cs*diff_cs
                                    delta_cs=cslw_nca[i1][3]**2+ cslw_nco [i2]**2
                                    delta_d=delta_cs-diff_cs
                                    if delta_d >= 0:
                                        cd_nmatch=cd_nmatch+1
                                        cd_possmatch.append(i2)

                                    if ngoodfreq_nca[i1]-4 >0:
                                        diff_cs=csfreq_nca[i1][4]-csfreq_nco[i2]
                                        diff_cs=diff_cs*diff_cs
                                        delta_cs=cslw_nca[i1][4]**2+ cslw_nco [i2]**2
                                        delta_e=delta_cs-diff_cs
                                        if delta_e >= 0:
                                            ce_nmatch=ce_nmatch+1
                                            ce_possmatch.append(i2)
                ##################You need to compare each sig in nco with each site in csfreq_nca
                    #if i1==2:
                        #print('i2 nca ngoodfreq_nca is', ngoodfreq_nca[i1])
                        #print('check now',possmatch[i1])
                if ngoodfreq_nca[i1] == 1:
                    if ca_nmatch>0:
                        nmatch[i1] = ca_nmatch

                        for k1 in range(1,ca_nmatch+1):
                            #nmatch[i1]= nmatch[i1]+1 #ue dont need to increment nmatch since only one carbon site to be matched
                            possmatch[i1].append([ca_possmatch[k1]])
                            fix[i1].append([0])
                            overuse[i1].append([[0]])
                            kca_nco=ca_possmatch[k1]
                            used_nco[kca_nco]+=1


                if ngoodfreq_nca[i1] == 2:
                    if ca_nmatch*cb_nmatch>0:
                        for k1 in range(1,ca_nmatch+1):
                            for k2 in range(1, cb_nmatch+1):
                                #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                                diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                                diff_n1=diff_n1**2
                                diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                                diff_nlw1=diff_nlw1*disparity_nco1**2

                                if diff_n1<=diff_nlw1:
                                    #their CO freq must also align
                                    diff_co1=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cb_possmatch[k2]]
                                    diff_co1=diff_co1**2
                                    diff_colw1=clw_nco[ca_possmatch[k1]]**2+clw_nco[cb_possmatch[k2]]**2
                                    diff_colw1=diff_colw1*disparity_nco2**2
                                    if diff_co1<=diff_colw1:
                                        nmatch[i1]= nmatch[i1]+1
                                        possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2]])
                                        overuse[i1].append([[0]])
                                        fix[i1].append([0])
                                        kca_nco=ca_possmatch[k1]
                                        used_nco[kca_nco]+=1
                                        kcb_nco=cb_possmatch[k2]
                                        used_nco[kcb_nco]+=1


                if ngoodfreq_nca[i1] == 3:
                    if ca_nmatch*cb_nmatch*cg_nmatch>0:
                        for k1 in range(1,ca_nmatch+1):
                            for k2 in range(1, cb_nmatch+1):
                                #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                                diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                                diff_n1=diff_n1**2
                                diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                                diff_nlw1=diff_nlw1*disparity_nco1**2
                                if diff_n1<=diff_nlw1:
                                    diff_co1=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cb_possmatch[k2]]
                                    diff_co1=diff_co1**2
                                    diff_colw1=clw_nco[ca_possmatch[k1]]**2+clw_nco[cb_possmatch[k2]]**2
                                    diff_colw1=diff_colw1*disparity_nco2**2
                                    if diff_co1<=diff_colw1:
                                        for k3 in range(1, cg_nmatch+1):
                                            diff_n2=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cg_possmatch[k3]]
                                            diff_n2=diff_n2**2
                                            diff_nlw2=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cg_possmatch[k3]]**2
                                            diff_nlw2=diff_nlw2*disparity_nco1**2
                                            #their CO freq must also align
                                            if diff_n2<=diff_nlw2:
                                                diff_co2=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cg_possmatch[k3]]
                                                diff_co2=diff_co2**2
                                                diff_colw2=clw_nco[ca_possmatch[k1]]**2+clw_nco[cg_possmatch[k3]]**2
                                                diff_colw2=diff_colw2*disparity_nco2**2
                                                if diff_co2<=diff_colw2:
                                                    nmatch[i1]= nmatch[i1]+1
                                                    possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2],cg_possmatch[k3]])
                                                    fix[i1].append([0])
                                                    overuse[i1].append([[0]])
                                                    kca_nco=ca_possmatch[k1]
                                                    used_nco[kca_nco]+=1
                                                    kcb_nco=cb_possmatch[k2]
                                                    used_nco[kcb_nco]+=1
                                                    kcg_nco=cg_possmatch[k3]
                                                    used_nco[kcg_nco]+=1


                if ngoodfreq_nca[i1] == 4:
                    if ca_nmatch*cb_nmatch*cg_nmatch*cd_nmatch>0:
                        for k1 in range(1,ca_nmatch+1):
                            for k2 in range(1, cb_nmatch+1):
                                #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                                diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                                diff_n1=diff_n1**2
                                diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                                diff_nlw1=diff_nlw1*disparity_nco1**2
                                if diff_n1<=diff_nlw1:
                                    diff_co1=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cb_possmatch[k2]]
                                    diff_co1=diff_co1**2
                                    diff_colw1=clw_nco[ca_possmatch[k1]]**2+clw_nco[cb_possmatch[k2]]**2
                                    diff_colw1=diff_colw1*disparity_nco2**2
                                    if diff_co1<=diff_colw1:
                                        for k3 in range(1, cg_nmatch+1):
                                            diff_n2=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cg_possmatch[k3]]
                                            diff_n2=diff_n2**2
                                            diff_nlw2=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cg_possmatch[k3]]**2
                                            diff_nlw2=diff_nlw2*disparity_nco1**2
                                            #their CO freq must also align
                                            if diff_n2<=diff_nlw2:
                                                diff_co2=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cg_possmatch[k3]]
                                                diff_co2=diff_co2**2
                                                diff_colw2=clw_nco[ca_possmatch[k1]]**2+clw_nco[cg_possmatch[k3]]**2
                                                diff_colw2=diff_colw2*disparity_nco2**2
                                                if diff_co2<=diff_colw2:
                                                    for k4 in range(1, cd_nmatch+1):
                                                        diff_n3=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cd_possmatch[k4]]
                                                        diff_n3=diff_n3**2
                                                        diff_nlw3=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cd_possmatch[k4]]**2
                                                        diff_nlw3=diff_nlw3*disparity_nco1**2
                                                        if diff_n3<=diff_nlw3:
                                                            diff_co3=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cd_possmatch[k4]]
                                                            diff_co3=diff_co3**2
                                                            diff_colw3=clw_nco[ca_possmatch[k1]]**2+clw_nco[cd_possmatch[k4]]**2
                                                            diff_colw3=diff_colw3*disparity_nco2**2
                                                            if diff_co3<=diff_colw3:
                                                                nmatch[i1]= nmatch[i1]+1
                                                                possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2],cg_possmatch[k3],cd_possmatch[k4]])
                                                                fix[i1].append([0])
                                                                overuse[i1].append([[0]])
                                                                kca_nco=ca_possmatch[k1]
                                                                used_nco[kca_nco]+=1
                                                                kcb_nco=cb_possmatch[k2]
                                                                used_nco[kcb_nco]+=1
                                                                kcg_nco=cg_possmatch[k3]
                                                                used_nco[kcg_nco]+=1
                                                                kcd_nco=cd_possmatch[k4]
                                                                used_nco[kcd_nco]+=1


                if ngoodfreq_nca[i1] == 5:
                    if ca_nmatch*cb_nmatch*cg_nmatch*cd_nmatch*ce_nmatch>0:
                        for k1 in range(1,ca_nmatch+1):
                            for k2 in range(1, cb_nmatch+1):
                                #if there are multiple carbon sites(except CO),we need to inspect these peaks also appear on the same nitrogen plane in ncocx
                                diff_n1=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cb_possmatch[k2]]
                                diff_n1=diff_n1**2
                                diff_nlw1=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cb_possmatch[k2]]**2
                                diff_nlw1=diff_nlw1*disparity_nco1**2
                                if diff_n1<=diff_nlw1:
                                    diff_co1=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cb_possmatch[k2]]
                                    diff_co1=diff_co1**2
                                    diff_colw1=clw_nco[ca_possmatch[k1]]**2+clw_nco[cb_possmatch[k2]]**2
                                    diff_colw1=diff_colw1*disparity_nco2**2
                                    if diff_co1<=diff_colw1:
                                        for k3 in range(1, cg_nmatch+1):
                                            diff_n2=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cg_possmatch[k3]]
                                            diff_n2=diff_n2**2
                                            diff_nlw2=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cg_possmatch[k3]]**2
                                            diff_nlw2=diff_nlw2*disparity_nco1**2
                                            #their CO freq must also align
                                            if diff_n2<=diff_nlw2:
                                                diff_co2=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cg_possmatch[k3]]
                                                diff_co2=diff_co2**2
                                                diff_colw2=clw_nco[ca_possmatch[k1]]**2+clw_nco[cg_possmatch[k3]]**2
                                                diff_colw2=diff_colw2*disparity_nco2**2
                                                if diff_co2<=diff_colw2:
                                                    for k4 in range(1, cd_nmatch+1):
                                                        diff_n3=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[cd_possmatch[k4]]
                                                        diff_n3=diff_n3**2
                                                        diff_nlw3=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[cd_possmatch[k4]]**2
                                                        diff_nlw3=diff_nlw3*disparity_nco1**2
                                                        if diff_n3<=diff_nlw3:
                                                            diff_co3=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[cd_possmatch[k4]]
                                                            diff_co3=diff_co3**2
                                                            diff_colw3=clw_nco[ca_possmatch[k1]]**2+clw_nco[cd_possmatch[k4]]**2
                                                            diff_colw3=diff_colw3*disparity_nco2**2
                                                            if diff_co3<=diff_colw3:
                                                                for k5 in range(1, ce_nmatch+1):
                                                                    diff_n4=n15freq_nco[ca_possmatch[k1]]-n15freq_nco[ce_possmatch[k5]]
                                                                    diff_n4=diff_n4**2
                                                                    diff_nlw4=n15lw_nco[ca_possmatch[k1]]**2+n15lw_nco[ce_possmatch[k5]]**2
                                                                    diff_nlw4=diff_nlw4*disparity_nco1**2
                                                                    if diff_n4<=diff_nlw4:
                                                                        diff_co4=cfreq_nco[ca_possmatch[k1]]-cfreq_nco[ce_possmatch[k5]]
                                                                        diff_co4=diff_co4**2
                                                                        diff_colw4=clw_nco[ca_possmatch[k1]]**2+clw_nco[ce_possmatch[k5]]**2
                                                                        diff_colw4=diff_colw4*disparity_nco2**2
                                                                        if diff_co3<=diff_colw3:
                                                                            nmatch[i1]= nmatch[i1]+1
                                                                            possmatch[i1].append([ca_possmatch[k1],cb_possmatch[k2],cg_possmatch[k3],cd_possmatch[k4],ce_possmatch[k5]])
                                                                            fix[i1].append([0])
                                                                            overuse[i1].append([[0]])
                                                                            kca_nco=ca_possmatch[k1]
                                                                            used_nco[kca_nco]+=1
                                                                            kcb_nco=cb_possmatch[k2]
                                                                            used_nco[kcb_nco]+=1
                                                                            kcg_nco=cg_possmatch[k3]
                                                                            used_nco[kcg_nco]+=1
                                                                            kcd_nco=cd_possmatch[k4]
                                                                            used_nco[kcd_nco]+=1
                                                                            kce_nco=ce_possmatch[k5]
                                                                            used_nco[kce_nco]+=1

end_time = time.time()
elapsed_time = end_time - start_time
print('ncocx signal table match to ncacx residue type assignments completes')
print('matching ncocx to ncacx took ', elapsed_time,' seconds.')
print('\n')
#fill fictitious points for 0th entry of possmatch

###############################################################################
#now sort out the used vs deg situation of matched signals for ncacx
print('now collect the statistics of the match results.')
for i1 in range(1,npeak_nca+1):
    if nca_fix[i1] == 1: #if this nca is a definite asgn
        #print('nca row ',i,' has bound value :', bound[i])
        if nmatch[i1]>0:
            flag=1
            neg_pair = 0
            temp_overuse=[]
            for match_id in range(1,nmatch[i1]+1):
                match_len=len(possmatch[i1][match_id])
                for i2 in range(0,match_len):
                    ksite=possmatch[i1][match_id][i2]
                    if deg_nco[ksite]<used_nco[ksite]:
                        flag = -1
                        neg_pair -= 1
                        temp_overuse.append(ksite)
                fix[i1][match_id] = neg_pair

                if flag >= 0:
                    fix[i1][match_id]=1
                else: #there are nco signal double dipping here in the match

                    overuse[i1][match_id]=temp_overuse
            #print('usage for nca row ',i,' match nco site is ', used_nco[ksite])
for i1 in range(1, npeak_nca+1):
    if nmatch[i1]>0:
        for match_id in range(1,nmatch[i1]+1):
            flag=1
            neg_pair = 0
            temp_overuse=[]
            match_len=len(possmatch[i1][match_id])
            for i3 in range(0,match_len):
                #print('nca row ',i1,' has bound value :', bound[i1])
                ksite=possmatch[i1][match_id][i3]
                #print('ksite is ',ksite)
                if deg_nco[ksite]<used_nco[ksite]:
                    flag = -1
                    neg_pair -= 1
                    temp_overuse.append(ksite)
            fix[i1][match_id] = neg_pair

            if flag >= 0:

                fix[i1][match_id]=1

            else: #there are nco signal double dipping here in the match

                overuse[i1][match_id]=temp_overuse
                #bound[i1]=-1
                #print('nca asgn is ', i1)

                    #print('ncacx row ',i1,' shows a non-overlapping nco at matching number ',i2)
            #if flag < 0:
                #print('ncacx row ',i1,' shows a overlapping nco at matching number ',i2)

###############################################################################
#end of match nco to nca
###############################################################################
#sort out nca and nco pairs with no concerns of local minimums
priority = [1 for i in range(npeak_nca+1)]#equals 1 for those nca plus pairing nco without concern for local minimums
overlap =[0 for i in range(npeak_nca+1)]
for i1 in range(npeak_nca+1):
    #tempn15freq_nca1 = n15freq_nca[i1]
    #tempn15lw_nca1 = n15lw_nca[i1]

    temp_num = extract_number(rtyp_nca[i1])
    if len(temp_num) > 0 :#definitely asgned should be excluded.
        priority[i1] = 2
    if nmatch[i1] >0 :
        temp_rtyp1 = extract_letter(rtyp_nca[i1])
        for i2 in range(i1+1,npeak_nca+1):
            temp_num = extract_number(rtyp_nca[i2])#definitely asgned should be excluded.
            if len(temp_num) > 0 :#definitely asgned should be excluded.
                priority[i2] = 2

            #if priority[i1] == 1:

            temp_rtyp2 = extract_letter(rtyp_nca[i2])
            if rtyp_compare(temp_rtyp1,temp_rtyp2) > 0:#this is to include consideration of ambiguous asgn residue types

                if temp_rtyp1 == 'G' :
                    if csfreq_nca[i1][0] != 1e6:
                        diff_freq1 = csfreq_nca[i1][0] - csfreq_nca[i2][0]
                        diff_freq1 *= diff_freq1
                        if cfreq_nca[i1] < 200.0 and cfreq_nca[i2] < 200.0: #they must have valid c' asgn
                            diff_freq2 = cfreq_nca[i1] - cfreq_nca[i2]
                            diff_freq2 *= diff_freq2

                            diff_freq = diff_freq1 + diff_freq2

                            if diff_freq > delta_priority * delta_priority:

                                flag = 0
                                if nmatch[i2] > 0 :
                                    diff_freq = n15freq_nca[i1] - n15freq_nca[i2]
                                    diff_freq *= diff_freq
                                    delta_lw = (n15lw_nca[i1] + n15lw_nca[i2])**2 # not sqrt, so to make sure the range will make a difference#(n15lw_nca[i1])**2 + (n15lw_nca[i2])**2
                                    if diff_freq <= delta_lw:
                                        #priority[i1] = 0
                                        #priority[i2] = 0
                                        #overlap[i1] += 1
                                        #overlap[i2] += 1
                                        #flag1 = 1

                                        for j1 in range(1,nmatch[i1]+1):
                                            #if priority[i1] == 1:
                                            temp_nco1 = possmatch[i1][j1][0]
                                            tempn15freq_nco1 = n15freq_nco[temp_nco1]
                                            tempn15lw_nco1 = n15lw_nco[temp_nco1]
                                            for j2 in range(1,nmatch[i2]+1):
                                                #if priority[i1] == 1:
                                                temp_nco2 = possmatch[i2][j2][0]
                                                tempn15freq_nco2 = n15freq_nco[temp_nco2]
                                                tempn15lw_nco2 = n15lw_nco[temp_nco2]
                                                diff_freq = tempn15freq_nco1 - tempn15freq_nco2
                                                diff_freq *= diff_freq
                                                #delta_lw = (tempn15lw_nco1)**2 + (tempn15lw_nco2)**2
                                                delta_lw = (tempn15lw_nco1 + tempn15lw_nco2)**2 # not sqrt, so to make sure the range will make a difference#
                                                if diff_freq <= delta_lw:
                                                    priority[i1] = 0
                                                    priority[i2] = 0
                                                    #overlap[i1] += 1
                                                    #overlap[i2] += 1
                                                    if flag == 0: #overlap with the same sets of nca pair nco is only counted once.
                                                        overlap[i1] += 1
                                                        overlap[i2] += 1
                                                        flag = 1

                        else:
                            if nmatch[i2] > 0 :
                                flag = 0
                                diff_freq = n15freq_nca[i1] - n15freq_nca[i2]
                                diff_freq *= diff_freq
                                delta_lw = (n15lw_nca[i1] + n15lw_nca[i2])**2 # not sqrt, so to make sure the range will make a difference#(n15lw_nca[i1])**2 + (n15lw_nca[i2])**2
                                if diff_freq <= delta_lw:
                                    #priority[i1] = 0
                                    #priority[i2] = 0
                                    #overlap[i1] += 1
                                    #overlap[i2] += 1
                                    #flag = 1

                                    for j1 in range(1,nmatch[i1]+1):
                                        #if priority[i1] == 1:
                                        temp_nco1 = possmatch[i1][j1][0]
                                        tempn15freq_nco1 = n15freq_nco[temp_nco1]
                                        tempn15lw_nco1 = n15lw_nco[temp_nco1]
                                        for j2 in range(1,nmatch[i2]+1):
                                            #if priority[i1] == 1:
                                            temp_nco2 = possmatch[i2][j2][0]
                                            tempn15freq_nco2 = n15freq_nco[temp_nco2]
                                            tempn15lw_nco2 = n15lw_nco[temp_nco2]
                                            diff_freq = tempn15freq_nco1 - tempn15freq_nco2
                                            diff_freq *= diff_freq
                                            #delta_lw = (tempn15lw_nco1)**2 + (tempn15lw_nco2)**2
                                            delta_lw = (tempn15lw_nco1 + tempn15lw_nco2)**2 # not sqrt, so to make sure the range will make a difference#

                                            if diff_freq <= delta_lw:
                                                priority[i1] = 0
                                                priority[i2] = 0
                                                if flag == 0:
                                                    overlap[i1] += 1
                                                    overlap[i2] += 1
                                                    flag = 1
                #elif nmatch[i1] == 0 :

                if temp_rtyp1 != 'G' :
                    if csfreq_nca[i1][0] != 1e6:
                        diff_freq1 = csfreq_nca[i1][0] - csfreq_nca[i2][0]
                        diff_freq1 *= diff_freq1
                        if csfreq_nca[i1][1] != 1e6: #they must have valid cbeta asgn
                            diff_freq2 = csfreq_nca[i1][1]- csfreq_nca[i2][1]
                            diff_freq2 *= diff_freq2

                            diff_freq = diff_freq1 + diff_freq2

                            if diff_freq > delta_priority * delta_priority:


                                if nmatch[i2] > 0 :
                                    flag = 0
                                    diff_freq = n15freq_nca[i1] - n15freq_nca[i2]
                                    diff_freq *= diff_freq
                                    delta_lw = (n15lw_nca[i1] + n15lw_nca[i2])**2 # not sqrt, so to make sure the range will make a difference#(n15lw_nca[i1])**2 + (n15lw_nca[i2])**2
                                    if diff_freq <= delta_lw:
                                        #priority[i1] = 0
                                        #priority[i2] = 0
                                        #overlap[i1] += 1
                                        #overlap[i2] += 1
                                        #flag = 1
                                        for j1 in range(1,nmatch[i1]+1):
                                            #if priority[i1] == 1:
                                            temp_nco1 = possmatch[i1][j1][0]
                                            tempn15freq_nco1 = n15freq_nco[temp_nco1]
                                            tempn15lw_nco1 = n15lw_nco[temp_nco1]
                                            for j2 in range(1,nmatch[i2]+1):
                                                #if priority[i1] == 1:
                                                temp_nco2 = possmatch[i2][j2][0]
                                                tempn15freq_nco2 = n15freq_nco[temp_nco2]
                                                tempn15lw_nco2 = n15lw_nco[temp_nco2]
                                                diff_freq = tempn15freq_nco1 - tempn15freq_nco2
                                                diff_freq *= diff_freq
                                                #delta_lw = (tempn15lw_nco1)**2 + (tempn15lw_nco2)**2
                                                delta_lw = (tempn15lw_nco1 + tempn15lw_nco2)**2 # not sqrt, so to make sure the range will make a difference#

                                                if diff_freq <= delta_lw:
                                                    priority[i1] = 0
                                                    priority[i2] = 0
                                                    if flag == 0:
                                                        overlap[i1] += 1
                                                        overlap[i2] += 1
                                                        flag = 1

                        else:

                            if nmatch[i2] > 0 :
                                flag = 0
                                diff_freq = n15freq_nca[i1] - n15freq_nca[i2]
                                diff_freq *= diff_freq
                                delta_lw = (n15lw_nca[i1] + n15lw_nca[i2])**2 # not sqrt, so to make sure the range will make a difference#(n15lw_nca[i1])**2 + (n15lw_nca[i2])**2
                                if diff_freq <= delta_lw:
                                    #priority[i1] = 0
                                    #priority[i2] = 0
                                    #overlap[i1] += 1
                                    #overlap[i2] += 1
                                    #flag = 1
                                    for j1 in range(1,nmatch[i1]+1):
                                        #if priority[i1] == 1:
                                        temp_nco1 = possmatch[i1][j1][0]
                                        tempn15freq_nco1 = n15freq_nco[temp_nco1]
                                        tempn15lw_nco1 = n15lw_nco[temp_nco1]
                                        for j2 in range(1,nmatch[i2]+1):
                                            #if priority[i1] == 1:
                                            temp_nco2 = possmatch[i2][j2][0]
                                            tempn15freq_nco2 = n15freq_nco[temp_nco2]
                                            tempn15lw_nco2 = n15lw_nco[temp_nco2]
                                            diff_freq = tempn15freq_nco1 - tempn15freq_nco2
                                            diff_freq *= diff_freq
                                            #delta_lw = (tempn15lw_nco1)**2 + (tempn15lw_nco2)**2
                                            delta_lw = (tempn15lw_nco1 + tempn15lw_nco2)**2 # not sqrt, so to make sure the range will make a difference#

                                            if diff_freq <= delta_lw:
                                                priority[i1] = 0
                                                priority[i2] = 0
                                                if flag == 0:
                                                    overlap[i1] += 1
                                                    overlap[i2] += 1
                                                    flag = 1

priority_num = 0
for i1 in range(npeak_nca+1):
    #if overlap[i1] < 2:#those nca+nco pair with only one side overlap will not propogate
        #priority[i1] = 1
    if priority[i1] == 1:
          priority_num += 1

with open(overlap_filename, mode='w') as f:
    with redirect_stdout(f):
        for i1 in range(npeak_nca+1):
            print(overlap[i1])
#print('There are ',priority_num,' nca asgn (paired at least with one of its nco) are free from local minimums.')
#with open(priority_filename, mode='w') as f:
    #with redirect_stdout(f):
        #print('There are ',priority_num,' nca asgn (paired at least with one of its nco) are free from local minimums.')

#if rigor > 0 :
    #with open(priority_filename, mode='w') as f:
       # with redirect_stdout(f):
           # print('There are ',priority_num,' nca asgn (paired at least with one of its nco) are free from local minimums.')
            #for i in range(npeak_nca+1):
               # print(priority[i])

#if rigor == 0 :
    #def_asgn = 0
    #priority_num2 = 0
    #for i in range(npeak_nca+1):
        #if priority[i] == 0:
            #x1 = random.random()
            #if x1 >= 0.5:
               # priority[i] = 1

       # if priority[i] == 1:
              #priority_num2 += 1
        #elif priority[i] == 2:
              #def_asgn += 1

    #print('There are ',def_asgn,' nca definitely asgned, and there are ',priority_num,' nca asgn (paired at least with one of its nco) are free from local minimums.')
    #print('There are additional ',priority_num2-priority_num,' randomly assign priority to those not defintely asgned, since there are not enough asgns are free from local minimums')
    #with open(priority_filename, mode='w') as f:
        #with redirect_stdout(f):
           # print('There are ',def_asgn,' nca definitely asgned, and there are ',priority_num,' nca asgn (paired at least with one of its nco) are free from local minimums.')
           # print('There are additional ',priority_num2-priority_num,' randomly assign priority to those not defintely asgned, since there are not enough asgns are free from local minimums')
           # for i in range(npeak_nca+1):
               # print(priority[i])




###############################################################################
#plot the summary of the ncocx2ncacx match
#matplotlib.use('Qt5Agg')

# set width of bar
res_index = []
protein_numseq = [0]
nca_index = []
defasgn_nca = [0 for i in range(nres+1)]
aa_names = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
aa_str = []
aa_ct = [0 for i in range(len(aa_names))]
ncares_ct = [0 for i in range(len(aa_names))]
ncaasgned_ct = [0 for i in range(len(aa_names))] #number of definitedly asgned nca in each residue type
errorfree_ct = [0 for i in range(len(aa_names))] #number of asgn free from local minimum concerns in each residue type
#res in protein nca has no CO freq specified in the protein
noco_nresct = [1 for i in range(npeak_nca+1)]
#number of unique match count for each residue type in the protein
unique_nresct = [0 for i in range(len(aa_names))]
#number of unique match count for each nca row
unique_ncact = [0 for i in range(npeak_nca+1)]
#number of overused match count for each residue type in the protein
overuse_nresct = [0 for i in range(len(aa_names))]
#if a nca row has overused match
overuse_ncact = [0 for i in range(npeak_nca+1)]
#number of total match count for each residue type in the protein
totmatch_nresct = [0 for i in range(len(aa_names))]
#number of total match count for each nca row
totmatch_ncact =[0 for i in range(npeak_nca+1)]
#number of total match count for each nca row for the nca with CO freq input
totmatchwCO_ncact =[0 for i in range(npeak_nca+1)]
#number of total match count for each nca row for the nca without CO freq input
totmatchwoCO_ncact =[0 for i in range(npeak_nca+1)]
#number of total none overlaping match count for each residue type in the protein
totnolmatch_nresct = [0 for i in range(len(aa_names))]
#number of total none overlapping match count for each nca row
totnolmatch_ncact =[0 for i in range(npeak_nca+1)]
#number of total overlaping match count for each residue type in the protein
totolmatch_nresct = [0 for i in range(len(aa_names))]
#number of total overlapping match count for each nca row
totolmatch_ncact =[0 for i in range(npeak_nca+1)]
for i in range(len(aa_names)):
    aa_str.append(aa_names[i])
#print(aa_str)
#print(aa_str.index(krtyp))
for i in range(1,nres+1):
    res_index.append(i)
    krtyp = extract_letter(protein[i]) #recall the index 0 is empty str to hold place for protein
    #print(krtyp)
    k_index = aa_str.index(krtyp)

    aa_ct[k_index] += 1
    krtyp = krtyp + str(i)
    protein_numseq.append(krtyp)

    if nposs[i] == -1:
        defasgn_nca[i] = 1
    else:
        defasgn_nca[i] = 0



for i1 in range(1, npeak_nca+1):
    nca_index.append(i1)
    kmatch = nmatch[i1]
    krtyp_old= extract_letter(rtyp_nca[i1])
    cyc_len = len(krtyp_old)
    if cyc_len > 1:
        for k_temp in range(cyc_len):
            krtyp = krtyp_old[k_temp]
            k_aaindex = aa_str.index(krtyp)
            ncares_ct[k_aaindex] += 1
            if priority[i1] > 0 :
                errorfree_ct[k_aaindex] += 1
            if len(extract_number(rtyp_nca[i1])) > 0:
                ncaasgned_ct[k_aaindex] += 1
            totmatch_nresct[k_aaindex] += kmatch
            totmatch_ncact[i1] += kmatch
    else:
        k_aaindex = aa_str.index(krtyp_old)
        ncares_ct[k_aaindex] += 1
        totmatch_nresct[k_aaindex] += kmatch
        totmatch_ncact[i1] += kmatch
        if priority[i1] > 0 :
            errorfree_ct[k_aaindex] += 1
        if len(extract_number(rtyp_nca[i1])) > 0:
            ncaasgned_ct[k_aaindex] += 1


    if cfreq_nca[i1] < 200.0:
        totmatchwCO_ncact[i1] += kmatch
        noco_nresct[i1] = 0

    if cfreq_nca[i1] == 1e6:
        totmatchwoCO_ncact[i1] += kmatch


    if bound[i1] == 1:
        unique_nresct[k_aaindex] += 1
        unique_ncact[i1] += 1
    if bound[i1] == -1:
        overuse_nresct[k_aaindex] += 1
        overuse_ncact[i1] += 1

    if kmatch > 0:
        for i2 in range(1,kmatch):
            if fix[i1][i2] == 1:
                totnolmatch_nresct[k_aaindex] += 1
                totnolmatch_ncact[i1] += 1
            elif fix[i1][i2] == -1:
                totolmatch_nresct[k_aaindex] += 1
                totolmatch_ncact[i1] += 1

########################################################################
#assign random priority
#if rigor > 0 :
    #with open(priority_filename, mode='w') as f:
        #with redirect_stdout(f):
           # print('There are ',priority_num,' nca asgn (paired at least with one of its nco) are free from local minimums.')
            #for i in range(npeak_nca+1):
               # print(priority[i])

#if rigor == 0 :
def_asgn = 0
    #priority_num2 = 0
for i in range(npeak_nca+1):
    if priority[i] == 2:
        def_asgn += 1
          #  x1 = random.random()
           # if x1 >= 50.5: #we dont further bias those with local minimum concerns. Just perform long simulations
             #   priority[i] = 1

        #if priority[i] == 1:
           #   priority_num2 += 1
           #elif priority[i] == 2:
             # def_asgn += 1

print('There are ',def_asgn,' nca definitely asgned, and there are ',priority_num,' nca asgn (paired at least with one of its nco) are free from local minimums.')
#print('There are additional ',priority_num2-priority_num,' randomly assign priority to those not defintely asgned, since there are not enough asgns are free from local minimums')
with open(priority_filename, mode='w') as f:
    with redirect_stdout(f):
        print('There are ',def_asgn,' nca definitely asgned, and there are ',priority_num,' nca asgn (paired at least with one of its nco) are free from local minimums.')
        #print('There are additional ',priority_num2-priority_num,' randomly assign priority to those not defintely asgned, since there are not enough asgns are free from local minimums')
        for i in range(npeak_nca+1):
            print(priority[i])

#end of plotting nco match 2 nca
###############################################################################
#print out summary of the macth finding to file
###############################################################################

#original_stdout = sys.stdout
#bound=[0 for i in range(npeak_nca+1)]#bound[nca asgn]=1 if a nca has unique nco asgn and the used_nco of matched
#nco signals for this nca asgn is less than deg_nco of the signal.
match1=0
match2=0
match3=0
match4=0
match5=0
match0=0
fix_num=0
bound_num=0
nofix_num=0
freenco_ct=0
for i in range(1,nsig_nco+1):
    if used_nco[i] == 0:
        if csfreq_nco[i] != 1e6:
            freenco_ct += 1

#def_nca=0
###find out how many matched nco is for each nca asgn
for i1 in range(1,npeak_nca+1):
    #if nca_fix[i1] == 1:
        #def_nca += 1
    if nmatch[i1] == 1 :
        match1+=1
        #print('at ncacx row ',i1,', res type is ',rtyp_nca[i1])
        #print('nmatch is ',nmatch[i1])
        #print('fix[i1] is ',fix[i1])
        if fix[i1][1] == 1:
            #krtyp_nca=extract_letter(rtyp_nca[i1])
            #krtyp_nca=" ".join(re.findall("[a-zA-Z]+", krtyp_nca))#extract only the letter part of residue type asgn
            #if krtyp_nca in conf_res or cfreq_nca[i1] < 200.0:
            #if the residue type is one of the confident asg type, or CO freq in nca is used to find the nco match
                #bound[i1]=1
                bound_num+=1
            #else:
                #if nposs[i1] == -1:
                    #bound[i1]=1
                    #bound_num+=1
            #fix_num+=1
    elif nmatch[i1] == 2:
        match2+=1
        for i2 in range(1,nmatch[i1]+1):
            if fix[i1][i2] == 1:
                fix_num+=1
    elif nmatch[i1] == 3:
        match3+=1
        for i2 in range(1,nmatch[i1]+1):
            if fix[i1][i2] == 1:
                fix_num+=1
    elif nmatch[i1] == 4:
        match4+=1
        for i2 in range(1,nmatch[i1]+1):
            if fix[i1][i2] == 1:
                fix_num+=1
    elif nmatch[i1] == 5:
        match5+=1
        for i2 in range(1,nmatch[i1]+1):
            if fix[i1][i2] == 1:
                fix_num+=1
    else:
        if nmatch[i1] == 0:
            match0+=1
            for i2 in range(1,nmatch[i1]+1):
                if fix[i1][i2] == 1:
                    fix_num+=1
#print('check definite asgn now')
#for i in range(1, npeak_nca+1):
    #if nca_fix[i] == 1:
        #print('definite nca asgn row ',i,' has the fix matrix as: ',fix[i][1])
with open(match_filename, mode='w') as f:
    with redirect_stdout(f):
        print('Nco match to nca asgn search completed.')

        print('The protein amino acid sequence is:')

        print('The protein sequence read from ',protein_seq,' is: ')
        for count in range(len(protein_aa)):
            if count+1 < 10:
                print(protein_aa[count],end='')
            if count+1 == 10:
                print(protein_aa[count],'  ', end='')
            if count+1 > 10:
                remainder = (count+1)%10
                if remainder != 0:
                     print(protein_aa[count],end='')
                if remainder == 0:
                    if (count+1)%50 == 0:
                        print(protein_aa[count])
                    else:
                        print(protein_aa[count],'  ', end='')

        print('\n')
        print('The following is the breakdown of sequence composition and residue types in the input nca asignments.')
        #print('\n')
        print('amino acid types             num of copies in sequence         num of copies in ncacx input        num of definitely asgned in ncacx input        num of copies free from local minimum concerns in ncacx input')
        for i1 in range(len(aa_ct)):
            print(aa_names[i1],'          ', aa_ct[i1],'            ',ncares_ct[i1],'        ', ncaasgned_ct[i1],'        ',errorfree_ct[i1])

        #analyze the sequence composition
        print('\n')
        print('Input statistics:')

        print('There are ',def_nca,' sequentially assigned nca asgn pre-existing in ncacx input')
        print('There are additional ',npeak_nca-def_nca,' none definite nca asgn.')
        print('There are ',nsig_nco,' individual resonances(peaks) from ncocx input.')
        print('There are ',freenco_ct, 'individual nco resonances(peaks) not used for match finding.')
        print('\n')

        print('Matching results:')
        print('There are ',match0,' nca inputs do not have matched nco')
        #sys.stdout = original_stdout
        print('There are ',match1,'nca inputs have 1 unique matched nco' )

        print('There are ',match2,' nca inputs have 2 matched nco' )

        print('There are ',match3,' nca inputs have 3 matched nco' )
        print('There are ',match4,' nca inputs have 4 matched nco' )
        print('There are ',match5,' nca inputs have 5 matched nco' )
        print('There are ',bound_num,' nca inputs have non-overlaping and unique nco match.')
        print('There are additional ',fix_num,' nca inputs have non-overlaping and non unique nco match.')
        print('\n')
        print('Overlaping definition: if any nco signals used in a match is used more than ')
        print('its specified degeneracy values in the entire nco to nca match search process.')
        print('################################################################')
        if match0 == 0:
            print('All ncacx inputs have found at least one match')
        elif match0>0:
            print('There are ',match0,'ncacx inputs with no nco matches, and they are:')
            for i in range(1, npeak_nca+1):
                if nmatch[i] == 0:
                    print('ncacx row ',i,n15freq_nca[i],cfreq_nca[i],end='')
                    for j in range(0,nfreq_nca-2):
                        print(' ',csfreq_nca[i][j],end='')
                    print(' ',deg_nca[i],rtyp_nca[i])

        print('################################################################')
        if match1 >0:
            print('We find ', match1,' ncacx inputs have a unique matched nco residue type assignments:')
            for i in range(1, npeak_nca+1):
                if nmatch[i] == 1:
                    print('ncacx row ',i,n15freq_nca[i],cfreq_nca[i],end='')
                    for j in range(0,nfreq_nca-2):
                        print(' ',csfreq_nca[i][j],end='')
                    print(' ',deg_nca[i],rtyp_nca[i])

                    print('its nco match is ')
                    j=possmatch[i][1][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][1])):
                        match_id=possmatch[i][1][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

        print('################################################################')
        if match2 >0:
            print('We find ', match2,' ncacx inputs have 2 matched nco residue type assignments:')
            for i in range(1, npeak_nca+1):
                if nmatch[i] == 2:
                    print('ncacx row ',i,n15freq_nca[i],cfreq_nca[i],end='')
                    for j in range(0,nfreq_nca-2):
                        print(' ',csfreq_nca[i][j],end='')
                    print(' ',deg_nca[i],rtyp_nca[i])

                    print('its first nco match is ')
                    j=possmatch[i][1][0]
                    #print(len(possmatch[2]))
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][1])):
                        match_id=possmatch[i][1][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

                    print('its 2nd nco match is ')
                    j=possmatch[i][2][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][2])):
                        match_id=possmatch[i][2][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

        print('################################################################')
        if match3 >0:
            print('We find ', match3,' ncacx inputs have 3 matched nco residue type assignments:')
            for i in range(1, npeak_nca+1):
                if nmatch[i] == 3:
                    print('ncacx row ',i,n15freq_nca[i],cfreq_nca[i],end='')
                    for j in range(0,nfreq_nca-2):
                        print(' ',csfreq_nca[i][j],end='')
                    print(' ',deg_nca[i],rtyp_nca[i])

                    print('its first nco match is ')
                    j=possmatch[i][1][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][1])):
                        match_id=possmatch[i][1][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

                    print('its 2nd nco match is ')
                    j=possmatch[i][2][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][2])):
                        match_id=possmatch[i][2][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

                    print('its 3rd nco match is ')
                    j=possmatch[i][3][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][3])):
                        match_id=possmatch[i][3][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

        print('################################################################')
        if match4 >0:
            print('We find ', match4,' ncacx inputs have 4 matched nco residue type assignments:')
            for i in range(1, npeak_nca+1):
                if nmatch[i] == 4:
                    print('ncacx row ',i,n15freq_nca[i],cfreq_nca[i],end='')
                    for j in range(0,nfreq_nca-2):
                        print(' ',csfreq_nca[i][j],end='')
                    print(' ',deg_nca[i],rtyp_nca[i])

                    print('its first nco match is ')
                    j=possmatch[i][1][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][1])):
                        match_id=possmatch[i][1][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

                    print('its 2nd nco match is ')
                    j=possmatch[i][2][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][2])):
                        match_id=possmatch[i][2][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

                    print('its 3rd nco match is ')
                    j=possmatch[i][3][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][3])):
                        match_id=possmatch[i][3][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

                    print('its 4th nco match is ')
                    j=possmatch[i][4][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][4])):
                        match_id=possmatch[i][4][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

        print('################################################################')
        if match5 >0:
            print('We find ', match5,' ncacx inputs have five matched nco residue type assignments:')
            for i in range(1, npeak_nca+1):
                if nmatch[i] == 5:
                    print('ncacx row ',i,n15freq_nca[i],cfreq_nca[i],end='')
                    for j in range(0,nfreq_nca-2):
                        print(' ',csfreq_nca[i][j],end='')
                    print(' ',deg_nca[i],rtyp_nca[i])

                    print('its first nco match is ')
                    j=possmatch[i][1][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][1])):
                        match_id=possmatch[i][1][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

                    print('its 2nd nco match is ')
                    j=possmatch[i][2][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][2])):
                        match_id=possmatch[i][2][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

                    print('its 3rd nco match is ')
                    j=possmatch[i][3][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][3])):
                        match_id=possmatch[i][3][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

                    print('its 4th nco match is ')
                    j=possmatch[i][4][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][4])):
                        match_id=possmatch[i][4][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')

                    print('its 5th nco match is ')
                    j=possmatch[i][5][0]
                    print(n15freq_nco[j],cfreq_nco[j],end='')
                    for j in range(0,len(possmatch[i][5])):
                        match_id=possmatch[i][5][j]
                        print(' ',csfreq_nco[match_id],end='')
                    print('\n')
        print('################################################################')
        print('The following list the nco signals never used in match finding:')
        usednco_tot=0
        overusednco_tot=0
        notusednco_tot=0
        for i in range(1,nsig_nco+1):

            if used_nco[i] == 0:
                if csfreq_nco[i] != 1e6:
                    notusednco_tot += 1
                    print('nco signal index ',i,' from row number ',nco_row[i],' in the original ncocx input file is not used.')
                    print('Its coordn are ',n15freq_nco[i],cfreq_nco[i],csfreq_nco[i],rtyp_nco[i])
            elif used_nco[i] <= deg_nco[i]:
                usednco_tot += 1
        #print('In total, there are ',notusednco_tot,' nco resonances not used.')
        print('################################################################')
        print('The following list the nco signals used more than their given degeneracy values (overused) in match finding:')
        for i in range(1, nsig_nco+1):
            if used_nco[i] > deg_nco[i]:
                overusednco_tot += 1
                print('nco signal index ',i,' from row number ',nco_row[i],' in the original ncocx input file is overused.')
                print('Its degeneracy value is ', deg_nco[i],'. It was used ',used_nco[i],' times in match finding.')
                print('Its coordn are ',n15freq_nco[i],cfreq_nco[i],csfreq_nco[i],rtyp_nco[i])
        print('In summary, the nco signal usage for matching nca asgn is described below:')
        print('there are ',usednco_tot,' nco signals used for matching nca asgn within their specified degeneracy range.')
        print('there are ',notusednco_tot,' nco signals are never used to match any nca asgn.')
        print('there are ',overusednco_tot,' nco signals are used over their allowed degeneracy number to match nca asgns.')
###############################################################################
#print out summary of the macth finding to file
###############################################################################
with open(matchdetail_filename, mode='w') as f:
    with redirect_stdout(f):
        #print line by line match to each nca row, easy to do a systematic check for the entire nca input
        print('#################################################################################')
        print('Now print line by line nco mtach to each nca input row.')
        print('Each row starts with the row index number of intendend nca match target,')
        print('it then follows by the N15, CO and respective carbon resonances of nco signals forming the matched residue type assignments,')
        print('then followed by their corresponding linewidth specification.')
        print('The last two columns are the degeneracy and residue type notation.')
        print('Their values are taken from those of the ncocx signal matched to the Calpha site of that residue type assignment.')
        print('#################################################################################')
        for i in range(1, npeak_nca+1):
            if nmatch[i] > 0:
                for j in range(1, nmatch[i]+1):
                    ksite = possmatch [i][j][0]
                    print('Match for ncacx row ',i,n15freq_nco[ksite],cfreq_nco[ksite],end='')
                    for k in range(0,nfreq_nca-2):
                        if csfreq_nca[i][k] != 1e6:#if the nca freq input is valid
                            match_id = possmatch [i][j][k]
                            print(' ',csfreq_nco[match_id],end='')
                        else:
                            print('  1e6',end='' )
                        #next print the linewidth
                    print(' ',n15lw_nco[ksite],clw_nco[ksite],end='')
                    for k in range(0,nfreq_nca-2):
                        if csfreq_nca[i][k] != 1e6:#if the nca freq input is valid
                            match_id = possmatch [i][j][k]
                            print(' ',cslw_nco[match_id],end='')
                        else:
                            print('  1e6',end='' )

                    print(' ',deg_nco[ksite],rtyp_nco[ksite])

#dont forget to reset the usage value for nco signals.
for i in range(1,nsig_nco+1):
    used_nco[i] = 0
#end output the match ncocx to ncacx residue type asgn to file.
###############################################################################

###############################################################################
###############################################################################
#function to prepare triplet residue inputs for evaluation of number of gbe
#you need to specify the first argument, which is the center residue
def triplet_input(kres):
    global current_nca,current_nco,n15freq_nca,n15lw_nca,n15freq_nco,n15lw_nco
    global possmatch,asgnfreq_nca,asgnlw_nca,asgnfreq_nco,asgnlw_nco
    site0_nca = current_nca[kres-1]
    site1_nca = current_nca[kres]
    site2_nca = current_nca[kres+1]

    asgnfreq_nca[0] = n15freq_nca[site0_nca]
    asgnfreq_nca[1] = n15freq_nca[site1_nca]
    asgnfreq_nca[2] = n15freq_nca[site2_nca]

    asgnlw_nca[0] = n15lw_nca[site0_nca]
    asgnlw_nca[1] = n15lw_nca[site1_nca]
    asgnlw_nca[2] = n15lw_nca[site2_nca]

    if site0_nca != 0:
        if current_nco[kres-1] != 0:
            site0_nco = possmatch[site0_nca][current_nco[kres-1]][0]
            asgnfreq_nco[0] = n15freq_nco[site0_nco]
            asgnlw_nco[0] = n15lw_nco[site0_nco]
        else:
            asgnfreq_nco[0] = 1e6
            asgnlw_nco[0] = 1e6
    else:
        asgnfreq_nco[0] = 1e6
        asgnlw_nco[0] = 1e6

    if site1_nca != 0:
        if current_nco[kres] != 0:
            site1_nco = possmatch[site1_nca][current_nco[kres]][0]
            asgnfreq_nco[1] = n15freq_nco[site1_nco]
            asgnlw_nco[1] = n15lw_nco[site1_nco]
        else:
            asgnfreq_nco[1] = 1e6
            asgnlw_nco[1] = 1e6
    else:
        asgnfreq_nco[1] = 1e6
        asgnlw_nco[1] = 1e6

    if site2_nca != 0:
        if current_nco[kres+1] != 0:
            site2_nco = possmatch[site2_nca][current_nco[kres+1]][0]
            asgnfreq_nco[2] = n15freq_nco[site2_nco]
            asgnlw_nco[2] = n15lw_nco[site2_nco]
        else:
            asgnfreq_nco[2] = 1e6
            asgnlw_nco[2] = 1e6
    else:
        asgnfreq_nco[2] = 1e6
        asgnlw_nco[2] = 1e6
    #return asgnfreq_nca,asgnlw_nca,asgnfreq_nco,asgnlw_nco

#end of #function to prepare triplet residue inputs for evaluation of number of gbe
###############################################################################
#function to asgn definite nca and its nco of the center residue for new configuration
def asgn_defnca(kres,kres_nca):
    import random
    global current_nca,current_nco,poss,nmatch,possmatch
    global asgnfreq_nca,asgnlw_nca,asgnfreq_nco,asgnlw_nco
    global skip_mc,nused_tot,defncaco_ct,bound
    temp_nca = current_nca[kres]
    nca_asgn[temp_nca] = 0
    current_nca[kres] = kres_nca
    asgnfreq_nca[1]=n15freq_nca[kres_nca]
    asgnlw_nca[1]=n15lw_nca[kres_nca]
    #nused_tot+=1
    if bound[kres_nca] == 1:
        defncaco_ct+=1
        nused_tot+=2
        skip_mc[kres]=1
        current_nco[kres] = 1
        kres_nco=possmatch[kres_nca][1][0]
        asgnfreq_nco[1]=n15freq_nco[kres_nco]
        asgnlw_nco[1]=n15lw_nco[kres_nco]
    else:
        if nmatch[kres_nca] == 1:
           #if the res has definite asgn for both nca and nco
            current_nco[kres]=1
            nused_tot+=1
            skip_mc[kres]=1#mark to skip mc draw process.
            kres_nco=possmatch[kres_nca][1][0]
            asgnfreq_nco[1]=n15freq_nco[kres_nco]
            asgnlw_nco[1]=n15lw_nco[kres_nco]
        elif nmatch[kres_nca] == 0:
            asgnfreq_nco[1] = 1e6
            asgnlw_nco[1] = 1e6
            current_nco[kres] = 0
        elif nmatch[kres_nca] > 1 : #and bound[kres_nca] != 1:
            match_index = random.randint(1, nmatch[kres_nca])
            current_nco[kres]=match_index
            nused_tot+=1
            kres_nco=possmatch[kres_nca][match_index][0]
            asgnfreq_nco[1]=n15freq_nco[kres_nco]
            asgnlw_nco[1]=n15lw_nco[kres_nco]
        #elif nmatch[kres_nca] > 1 and bound[kres_nca] == 1:
            #current_nco[kres]=1
            #nused_tot+=1
            #defncaco_ct+=1
            #skip_mc[kres]=1#mark to skip mc draw process
            #kres_nco=possmatch[kres_nca][1][0]
            #asgnfreq_nco[1]=n15freq_nco[kres_nco]
            #asgnlw_nco[1]=n15lw_nco[kres_nco]
    #return asgnfreq_nca, asgnlw_nca, asgnfreq_nco, asgnlw_nco, current_nca, current_nco, skip_mc, nused_tot,defncaco_ct

#end of function update definite asgn nca with its nco of the center residue for new configuration
###############################################################################

#define random draw function
###############################################################################

ctx = DrawContext(
    nres=nres,
    probzero=probzero_i,
    current_nca=nca_asgn,                # confirmed as assignment tracker
    current_nco=[0]*(nres+2),            # likely still unassigned at this point; init zero
    nposs=nposs,
    poss=poss,
    nmatch=nmatch,
    possmatch=possmatch,
    used_nca=used_nca,
    deg_nca=deg_nca,
    used_nco=used_nco,
    deg_nco=deg_nco,
    fix=fix,
    overuse=overuse,
    skip_mc=skip_mc
)
print(nres)

# DrawContext already imported at the top

# Assume all required assignment arrays are already defined above this block:
# nres, probzero, current_nca, current_nco, nposs, poss, nmatch, possmatch,
# used_nca, deg_nca, used_nco, deg_nco, fix, overuse, skip_mc

ctx = DrawContext(
    nres=nres,
    probzero=probzero_i,
    current_nca=current_nca,
    current_nco=current_nco,
    nposs=nposs,
    poss=poss,
    nmatch=nmatch,
    possmatch=possmatch,
    used_nca=used_nca,
    deg_nca=deg_nca,
    used_nco=used_nco,
    deg_nco=deg_nco,
    fix=fix,
    overuse=overuse,
    skip_mc=skip_mc
)

N_ITER = 2_000

for _ in range(N_ITER):
    ctx.draw()

# At this point, ctx.current_nca and ctx.current_nco hold the final assignments.
# Optionally, output or analyze the result:
print("Final NCA assignments:", ctx.current_nca)
print("Final NCO assignments:", ctx.current_nco)


def draw():
    import random
    global nres,probzero,current_nca,current_nco,nposs,poss,nmatch,possmatch
    global used_nca,deg_nca,used_nco,deg_nco,fix
    global nused_oldnca,nused_oldnco,nused_newnca,nused_newnco,skip_mc
    global asgn_newnca,asgn_newnco,asgn_oldnca,asgn_oldnco
    global kres
    select_exit = 0

    while select_exit == 0:


        draw_kres = 1
        #print('I am before kres selction')
        while draw_kres == 1:



            kres=random.randint(1,nres)# Random integer between 1 and nres. Both ends included.
            if skip_mc[kres] == 0:
                if nposs[kres] > 0:#current position must have at least one nca possible asgn
                    draw_kres = 0
                    asgn_oldnca=current_nca[kres]#check what is the residue type assignment assigned to kres
                    if asgn_oldnca != 0:
                        nused_oldnca = 1
                    else:
                        nused_oldnca = 0
                        nused_oldnco = 0
                    asgn_oldnco=current_nco[kres]
                    if asgn_oldnco != 0 :
                        nused_oldnco = 1
                    else:
                        nused_oldnco = 0
        #print('kres selected is ',kres)

        x1 = random.random() #generate a random float between 0 and 1(0 included, 1 not included)
        if x1 < probzero:
            nused_newnca = 0 #asgn_newnca was initialized to be zero, no need to do x < probzero
            asgn_newnca = 0
            asgn_newnco = 0
            nused_newnco = 0
                #if nposs[kres]>0: #if selected kres has candidate nca asgn
        else:
            kposs_nca = random.randint(1, nposs[kres]) #select an int between 1 to nposs
            asgn_newnca = poss[kres][kposs_nca]
            nused_newnca = 1
            #at this step asgn_newnca can be valid or null
            #print('nca old and new are ',asgn_oldnca, asgn_newnca)

        if deg_nca[asgn_newnca] <= used_nca[asgn_newnca]:
                #print('nca differs and no overuse exit')
            asgn_newnca = 0
            asgn_newnco = 0
            nused_newnca = 0
            nused_newnco = 0
        else:
            if nmatch[asgn_newnca] == 0:#if nca has non zero nco match
                asgn_newnco = 0
                nused_newnco = 0
                #print('asgn_newnco is ',asgn_newnco)
            else:
                asgn_newnco = random.randint(1, nmatch[asgn_newnca]) #select an int between 1 to nmatch
                #print('asgn_newnco is ',asgn_newnco)
                if fix[asgn_newnca][asgn_newnco] < 0: #if any signal in nco match is overused
                    overuse_mark = 0
                    for i1 in range(0,len(overuse[asgn_newnca][asgn_newnco])):
                        ksite = overuse[asgn_newnca][asgn_newnco][i1]
                        if deg_nco[ksite] <= used_nco[ksite]:
                            overuse_mark = 1
                    if overuse_mark == 1:
                        nused_newnco = 0
                        asgn_newnco = 0
                    elif overuse_mark == 0:
                                #asgn_newnco = knewnco
                        nused_newnco = 1
                else:#if nco match has no overuse concern
                    nused_newnco = 1

            if asgn_newnca == asgn_oldnca and asgn_newnco == asgn_oldnco:


                select_exit = 0
            else:
                select_exit = 1




    #return nused_oldnca,nused_oldnco,nused_newnca,nused_newnco,kres,asgn_newnca,asgn_newnco,asgn_oldnca,asgn_oldnco
###############################################################################
#End of function draw

#Define function ngbe to compute num of good, bad, and edge asgn
###############################################################################
def ngbe(kres):
    global asgnfreq_nca, asgnlw_nca, asgnfreq_nco, asgnlw_nco,nres
    nngood=0
    nnbad=0
    nnedge=0

    for i in range (1,3):#only scan i=kres and i=kres+1, which are the index 1 and 2 of the input lists
        freq_nca=asgnfreq_nca[i]
        freq_nco=asgnfreq_nco[i-1]
        lw_nca=asgnlw_nca[i]
        lw_nco=asgnlw_nco[i-1]

        if freq_nca < 1e6:
            if freq_nco < 1e6:# If both nca and nco freq exists. This excludes the case when kres=1 and nres.
                diff_freq=freq_nca-freq_nco
                diff_freq=diff_freq*diff_freq
                delta_freq=lw_nca*lw_nca+lw_nco*lw_nco

                if diff_freq <= delta_freq:
                    nngood = nngood+1
                else:
                    nnbad = nnbad+1
                    #time.sleep(3)

            elif freq_nco == 1e6:
                if i == 1 and kres == 1:# exclude kres is the first residue in protein.
                    nnedge += 0
                else:
                    nnedge += 1

        elif freq_nca == 1e6:
            if freq_nco < 1e6:
                if i == 2 and kres == nres:# exclude kres is the last residue in protein
                    nnedge += 0
                else:
                    nnedge += 1

    return nngood,nnbad,nnedge
        #both are null assignment
###############################################################################
#End of function ngbe
###############################################################################
#count the number of nonzero elements in a list, excluding the first element
def num_nonzero(a):
    a_len = len(a)
    num = 0
    for i in range(1,a_len):
        if a[i] != 0:
            num += 1
    return num
###############################################################################

#Start main Monte Carlo program
###############################################################################
print('Start Monte Carlo process.')
with open(runrecord_filename, mode='w') as f:
    with redirect_stdout(f):

        print('The ncacx input list is ',ncacx_filename)
        print('The ncocx input list is ',ncocx_filename)
        print('The protein sequence file is ',protein_seq)
        print('The total runs are ',run_num)
        print('The diagnostics set up is ',diagnostics )
        print('The total annealing step is ',nstep)
        print('The terminator step is set to ',terminator)
        print('The scale value for annealing slope is ',scale)
        print('The number of MC attemps per annealing step is ',nattempt)
        print('The rigor is set to ',rigor)
        print('The penalty for overlapping residues are ',penalty)
        print('The disparity for 1st dimension in ncacx is ', disparity_nca1)
        print('The disparity for 2nd dimension in ncacx is ', disparity_nca2)
        print('The disparity for 1st dimension in ncocx is ', disparity_nco1)
        print('The disparity for 2nd dimension in ncocx is ', disparity_nco2)
        print('The uncertainty of 15N in ncacx is ',ncan15lw_scalar)
        print('The uncertainty of carboxylic carbon in ncacx is ',ncaclw_scalar)
        print('The uncertainty of noncarboxylic carbon in ncacx is ',ncac13lw_scalar)
        print('The uncertainty of 15N in ncocx is ',ncon15lw_scalar)
        print('The uncertainty of carboxylic carbon in ncocx is ',ncoclw_scalar)
        print('The uncertainty of noncarboxylic carbon in ncocx is ',ncoc13lw_scalar)


#import random
#import math
#each istep row, ipeak_nca col of instigator stores the mobility of corresponding nca asgn (ipeak_nca) at istep of MCSA
#zero-th col stores the total mobility of the system at that step


consistency = [1 for i in range(0, nres+2)]
consistency[0] = 0
consistency[nres+1] = 0
consistency_nca = [0 for i in range(0, npeak_nca+1)]
correct_asgn = 0
rtyp_nextrd = ['' for i in range(npeak_nca+1)]

if diagnostics == 1:
###########################################################################
#home registers where nca asgn was asgned to in sequence. i-th col records the residue position in sequence
#its value registers how many times a nca asgn (row index) was asgned to this residue positions
#0-th col of each row registers how many different residue position this nca asgn was asgned to.
#the npeak_nca+1 element of each row stores the total usage num in MCSA in the run
#neighbor registers for each k-th nca-asgn (stack index), who are asgned as its neighbor in the sequence.
#the value of each i,j position, is the 0-th col value of i-th nca-asgn * j-th nca_asgn * k-th nca_asgn in that MCSA run
    neighbor_t = [[[0 for i in range(npeak_nca+1)] for j in range(npeak_nca+1)] for k in range(nstep+1)]
    instigator = [[0 for i in range(npeak_nca+1)] for j in range(nstep+1)]
    occupancy_sum = [[0 for j in range(nres+1)] for k in range(nstep+1)]
    occupancy_step = [[[0 for i in range(npeak_nca+1)] for j in range(nres+1)] for k in range(nstep+1)]
    all_number = [[0 for i in range(4)] for j in range(nstep+1)] #store the num of good, bad, edge and total used along each mcsa step
###########################################################################
#silence the skipped residues
klen=len(silencer)
if klen > 0 :
    for i in range(klen):
        ksite = silencer[i]
        deg_nca[ksite] = -1 #so it will never be used in MC selection
rtyp4bk_nca = rtyp_nca.copy()
rtyp4bk_nco = rtyp_nco.copy()
used4bk_nca = used_nca.copy()
used4bk_nco = used_nco.copy()

for n in range(run_num):
    start_time=time.time()
    w1inc=(w1f-w1i)/nstep * scale
    w2inc=(w2f-w2i)/nstep * scale
    w3inc=(w3f-w3i)/nstep * scale
    w4inc=(w4f-w4i)/nstep * scale
    #####################################
    #refresh used and rtyp for each spectra
    rtyp_nca = rtyp4bk_nca.copy()
    rtyp_nco = rtyp4bk_nco.copy()
    used_nca = used4bk_nca.copy()
    used_nco = used4bk_nco.copy()
    istep=1
    ####################################################################################
    available = [i for i in range(1,nres+1)]#stores the residue index for those are not definitely assigned,
    #so we can more efficiently do MC draw
    ngood_tot=0
    ngood_tot_bias = 0
    nbad_tot=0
    nedge_tot=0
    nused_tot=0

    asgnfreq_nca=[0.0 for i in range(0,3)]
    asgnlw_nca=[0.0 for i in range(0,3)]
    asgnfreq_nco=[0.0 for i in range(0,3)]
    asgnlw_nco=[0.0 for i in range(0,3)]

    accept_ct=[0 for i in range(0,nres+1)]
    nca_ct=[0 for i in range(0,npeak_nca+1)]

    #initialize the asgn:
    defnca_ct=0
    defncaco_ct=0
    current_nca=[0 for i in range(nres+2)]
    current_nco=[0 for i in range(nres+2)]

# ← here, create the context:
    ctx = DrawContext(
      nres=nres,
      probzero=probzero_i,
      current_nca=current_nca,
      current_nco=current_nco,
      nposs=nposs,
      poss=poss,
      nmatch=nmatch,
      possmatch=possmatch,
      used_nca=used_nca,
      deg_nca=deg_nca,
      used_nco=used_nco,
      deg_nco=deg_nco,
      fix=fix,
      overuse=overuse,
      skip_mc=skip_mc
    )
    #good_nca=[0 for i in range(nres+2)]#stores the latest good asgn.
    #good_nco=[0 for i in range(nres+2)]
    skip_mc=[0 for i in range(nres+2)]#equal to 1 for res with definite nca and nco asgn
    for i1 in range(0,npeak_nca+1):
        used_nca[i1] = 0
        nca_asgn[i1] = 0

    for i1 in range(0, nsig_nco+1):
        used_nco[i1] = 0

    print()
    print('This is ',n,' MCSA run out of ',run_num,' total runs in the iteration.')
    print('Terminator is set to ',terminator)
    print('total number of used sigs are ', nused_tot)
    print('There are ',len(silencer),' nca asgns that are excluded from mc asgn.')

    if len(silencer) > 0:
        print('The excluded nca asgns are: ')
        for i in range(len(silencer)):
            print(' ',silencer[i],end='')
        print('')
    if n == 0:
        with open(runrecord_filename, mode='a') as f:
            with redirect_stdout(f):
                print()
                print('This is ',n,' MCSA run out of ',run_num,' total runs in the iteration.')
                print('Terminator is set to ',terminator)
                print('total number of used sigs are ', nused_tot)
                print('There are ',len(silencer),' nca asgns that are excluded from mc asgn.')

                if len(silencer) > 0:
                    print('The excluded nca asgns are: ')
                    for i in range(len(silencer)):
                        print(' ',silencer[i],end='')
                    print('')
                print('total number of used sigs are ', nused_tot)
                print('There are ',len(silencer),' nca asgns that are excluded from mc asgn.')
                if len(silencer) > 0:
                    print('The excluded nca asgns are: ')
                    for i in range(len(silencer)):
                        print(' ',silencer[i],end='')
                    print('')
    else:
        with open(runrecord_filename, mode='a') as f:
            with redirect_stdout(f):
                print()
                print('This is ',n,' MCSA run out of ',run_num,' total runs in the iteration.')
                print('total number of used sigs are ', nused_tot)
                print('There are ',len(silencer),' nca asgns that are excluded from mc asgn.')

                if len(silencer) > 0:
                    print('The excluded nca asgns are: ')
                    for i in range(len(silencer)):
                        print(' ',silencer[i],end='')
                    print('')
                print('total number of used sigs are ', nused_tot)
                print('There are ',len(silencer),' nca asgns that are excluded from mc asgn.')
                if len(silencer) > 0:
                    print('The excluded nca asgns are: ')
                    for i in range(len(silencer)):
                        print(' ',silencer[i],end='')
                    print('')

    for i1 in range(1, nres+1):
        #for i2 in range(1, npeak_nca+1):
        if nposs[i1] == -1:
                defnca_ct += 1
                kres = i1
                #call function to create triple nca and nco parameters to evaluate num of gbe
                triplet_input(kres)#,current_nca,current_nco,n15freq_nca,n15lw_nca,n15freq_nco,n15lw_nco,possmatch,asgnfreq_nca,asgnlw_nca,asgnfreq_nco,asgnlw_nco)
                #evaluate old configuration's num of gbe
                nngood_old,nnbad_old,nnedge_old = ngbe(kres)#asgnfreq_nca, asgnlw_nca,asgnfreq_nco, asgnlw_nco,kres,nres)
                #end evaluation of old configuration's num of gbe
                #if current_nca[kres] != 0:
                #now we bias the ngood
                if rigor > 0:
                    if overlap[current_nca[kres]] > 0:
                        temp_nngood_old = nngood_old / penalty#nmatch[current_nca[kres]]
                    else:
                        temp_nngood_old = nngood_old
                else:
                    temp_nngood_old = nngood_old

                #update the nca and nco of the center residue of the triplet
                kres_nca = poss [i1][1]
                #replace nca nco asgn, update nused_tot, skip_mc
                #and update the center residue asgn for asgnfreq/lw_nca/nco
                asgn_defnca(kres,kres_nca)#,current_nca,current_nco,poss,nmatch,possmatch,asgnfreq_nca,asgnlw_nca,asgnfreq_nco,asgnlw_nco,skip_mc,nused_tot,defncaco_ct)
                nca_asgn[kres_nca] = kres
                if bound[kres_nca] == 1: #if the definitely asgned nca has definitely asgned nco to go for the kres
                    available.remove(kres)
                #now evaluate the new configuration with definite nca asgn and its nco
                nngood_new,nnbad_new,nnedge_new = ngbe(kres)#asgnfreq_nca, asgnlw_nca, asgnfreq_nco, asgnlw_nco,kres,nres)
                #end of computing num of gbe for insertino of definite asgn
                if nnbad_new > 0:
                    print('bad asgn res position is ', kres)
                    print('bad asgn nca is ', kres_nca)
                temp_nngood_new = nngood_new
                #update tot num
                ngood_tot += nngood_new - nngood_old
                ngood_tot_bias += temp_nngood_new - temp_nngood_old
                nbad_tot += nnbad_new - nnbad_old
                nedge_tot += nnedge_new - nnedge_old

    #end of initialization
    draw_num = len(available)

    print('there are ',defnca_ct,'residues have definite asgn for nca.')
    print('there are ',defncaco_ct,'residues have definite asgn for both nca and nco.')
    print('there are ',ngood_tot,' good asgn.')
    print('there are ',ngood_tot_bias,' biased good asgn.')
    print('there are ',nbad_tot,' bad asgn.')
    print('there are ',nedge_tot,' egde asgn.')
    print('there are ',nused_tot,' used residue type asgn.')
    #finish initialization of asgn
    with open(runrecord_filename, mode='a') as f:
        with redirect_stdout(f):
            print('there are ',defnca_ct,'residues have definite asgn for nca.')
            print('there are ',defncaco_ct,'residues have definite asgn for both nca and nco.')
            print('there are ',ngood_tot,' good asgn.')
            print('there are ',ngood_tot_bias,' biased good asgn.')
            print('there are ',nbad_tot,' bad asgn.')
            print('there are ',nedge_tot,' egde asgn.')
            print('there are ',nused_tot,' used residue type asgn.')

    while istep <= nstep:
        #delta_e = math.log(istep)/nstep
        naccept=0
        npick=0
        w1=w1i+(istep-1)*w1inc
        w2=w2i+(istep-1)*w2inc
        w3=w3i+(istep-1)*w3inc
        w4=w4i+(istep-1)*w4inc

        while npick < nattempt:

            npick += 1
            probzero = probzero_i * (nused_tot)/(npeak_nca + nsig_nco)
            #w3 = (nused_tot)/(npeak_nca+nsig_nco)
            #select a random residue in protein kres, and asgn nca and nco to kres
            #print('I am before draw')
            #draw()

                # … inside your inner nattempt loop …

            ctx.draw()
            # ← you must pull everything out right here
            kres          = ctx.kres
            asgn_oldnca   = ctx.asgn_oldnca
            asgn_oldnco   = ctx.asgn_oldnco
            asgn_newnca   = ctx.asgn_newnca
            asgn_newnco   = ctx.asgn_newnco
            nused_oldnca  = ctx.nused_oldnca
            nused_oldnco  = ctx.nused_oldnco
            nused_newnca  = ctx.nused_newnca
            nused_newnco  = ctx.nused_newnco


            triplet_input(kres)

            # now it’s safe to do this:
            asgnfreq_nca[1] = n15freq_nca[asgn_newnca]


            kres = ctx.kres
            asgn_old = ctx.asgn_oldnca
            asgn_new = ctx.asgn_newnca
            assert 1 <= kres     <= nres,        "Bad residue index"
            assert 0 <= asgn_old <= npeak_nca,   "Bad old NCA index"
            assert 0 <= asgn_new <= npeak_nca,   "Bad new NCA index"        # ← extract kres here
            triplet_input(kres)      # now passes a defined variable

            #print('I am after draw')
            #triplet_input(kres)
            nngood_old,nnbad_old,nnedge_old = ngbe(kres)

            #now we bias the nngood by priority marking of nca asgn
            if rigor > 0:
                if overlap[asgn_oldnca] > 0:
                    temp_nngood_old = nngood_old / penalty #nmatch[current_nca[kres]]
                else:
                    temp_nngood_old = nngood_old
            else:
                temp_nngood_old = nngood_old

            asgnfreq_nca[1] = n15freq_nca[asgn_newnca]
            asgnlw_nca[1] = n15lw_nca[asgn_newnca]

            if asgn_newnca != 0:
                if asgn_newnco != 0:
                    ksite = possmatch[asgn_newnca][asgn_newnco][0]
                else:
                    ksite = 0
            else:
                ksite = 0

            asgnfreq_nco[1] = n15freq_nco[ksite]
            asgnlw_nco[1] = n15lw_nco[ksite]

            nngood_new, nnbad_new, nnedge_new = ngbe(kres)

            #now we bias the nngood by priority marking of nca asgn
            if rigor > 0:
                if overlap[asgn_newnca] > 0:
                    temp_nngood_new = nngood_new / penalty #nmatch[current_nca[kres]]
                else:
                    temp_nngood_new = nngood_new
            else:
                temp_nngood_new = nngood_new


            delta_good = nngood_new-nngood_old
            #use the biased nngood
            delta_good_bias = temp_nngood_new - temp_nngood_old
            delta_bad = nnbad_new - nnbad_old
            delta_edge = nnedge_new - nnedge_old
            delta_used = nused_newnca - nused_oldnca + nused_newnco - nused_oldnco

            score = w1*(delta_good_bias)-w2*(delta_bad)-w3*(delta_edge)+w4*(delta_used)

            x = random.random()
            if x <= math.exp(score):
                #print('istep is ',istep)
                accept_ct[kres] += 1
                nca_ct[asgn_newnca] += 1
                naccept += 1
                if kres != 0:
                    flag = 0

                    if diagnostics == 1:
                        neighbor_l = current_nca[kres-1]
                        neighbor_r = current_nca[kres+1]
                        neighbor_t[istep][asgn_newnca][neighbor_l] += 1#delta_e
                        neighbor_t[istep][asgn_newnca][neighbor_r] += 1#delta_e
                        instigator[istep][asgn_newnca] += 1
                        occupancy_sum[istep][kres] += 1
                        occupancy_step[istep][kres][asgn_newnca] += 1

                current_nca[kres] = asgn_newnca


                if asgn_oldnca != 0 :
                    nca_asgn[asgn_oldnca] = 0
                    used_nca[asgn_oldnca] -= 1

                if asgn_newnca != 0 :
                    nca_asgn[asgn_newnca] = kres
                    used_nca[asgn_newnca] += 1


                current_nco[kres] = asgn_newnco

                #if score > 0: #if the new move is a good asgn
                    #good_nca[kres] = asgn_newnca
                    #good_nco[kres] = asgn_newnco

                if asgn_oldnca != 0:
                    if asgn_oldnco != 0:
                        if fix[asgn_oldnca][asgn_oldnco] < 0: #if any signal in nco match is overused
                            for i1 in range(0,len(overuse[asgn_oldnca][asgn_oldnco])):
                                ksite=overuse[asgn_oldnca][asgn_oldnco][i1]
                                used_nco[ksite] -= 1

                if asgn_newnca != 0:
                    if asgn_newnco != 0:
                        if fix[asgn_newnca][asgn_newnco] < 0: #if any signal in nco match is overused
                            for i1 in range(0,len(overuse[asgn_newnca][asgn_newnco])):
                                ksite=overuse[asgn_newnca][asgn_newnco][i1]
                                used_nco[ksite] += 1



                ngood_tot += delta_good
                ngood_tot_bias += delta_good_bias
                nbad_tot += delta_bad
                nedge_tot += delta_edge
                nused_tot += delta_used
        print('')
        print('This is the ',istep,' step.')
        print('naccept is ',naccept)
        #print('npeak is ',npick)
                #print('istep is ',istep,', the score is ',score)
        print('acceptance rate is ',naccept/nattempt)
        print('number of good asgn is ',ngood_tot,',number of biased good asgn is ',ngood_tot_bias,',number of bad asgn is ',nbad_tot,', number of edge asgn is ',nedge_tot,', number of used signals is ',nused_tot)
        with open(runrecord_filename, mode='a') as f:
            with redirect_stdout(f):
                print('')
                print('This is the ',istep,' step.')
                print('naccept is ',naccept)
                print('acceptance rate is ',naccept/nattempt)
                print('number of good asgn is ',ngood_tot,',number of biased good asgn is ',ngood_tot_bias,',number of bad asgn is ',nbad_tot,', number of edge asgn is ',nedge_tot,', number of used signals is ',nused_tot)

        if diagnostics == 1:
            instigator[istep][0] = sum(instigator[istep]) - instigator[istep][0]  #total energy
            all_number[istep][0] += ngood_tot
            all_number[istep][1] += nbad_tot
            all_number[istep][2] += nedge_tot
            all_number[istep][3] += nused_tot
        istep+=1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Monte Carlo draws completes, and it took ', elapsed_time,' seconds.')
    with open(runrecord_filename, mode='a') as f:
        with redirect_stdout(f):
            print('Monte Carlo draws completes, and it took ', elapsed_time,' seconds.')
    #collect asgn informatio from each asgn run
    #if n == 0:
    miss = 0
    reference_asgn = [0 for i in range(nres+1)]
    for i in range(nres+1):
        reference_asgn[i] = current_nca[i]
        if current_nca[i] == 0 :
            consistency[i] = 0
            reference_asgn[i] = 0

        elif current_nco[i] == 0:
            consistency[i] = 0
            reference_asgn[i] = 0
            if nmatch[current_nca[i]] > 0:# count how many of those free from local minimums, with no nco match towards not consistently asgned
                if overlap[current_nca[i]] < 1 :
                      miss += 1

        if rigor == 3:#when rigor == 3, only those marked by non overlap count towards consistenly asgned.
            if overlap[current_nca[i]] > 0:
                consistency[i] = 0
                reference_asgn[i] = 0
        if rigor == 2: #only nmatch = 1 will be counted towards consistently asgned to seed the next round
            if nmatch[current_nca[i]] > 1: #only nmatch = 1 can seed next rd
                consistency[i] = 0
                reference_asgn[i] = 0
        if rigor == 1: #only asgn without entanglment will be counted towards consistently asgned to seed the next round
            if fix[current_nca[i]][current_nco[i]] < 0:#only those without entanglement of matched nco will seed next rd
                consistency[i] = 0
                reference_asgn[i] = 0


    if n == 0:
        with open(runsummary_filename, mode='w') as f:
            with redirect_stdout(f):
                print('run number index    total good asgn    total bad asgn   total edge asgn   total used asgn    with nco pair but nco not asgned')
                print(n,'                     ',ngood_tot,'       ',nbad_tot,'      ',nedge_tot,'     ',nused_tot,'     ',miss)
    else:
        with open(runsummary_filename, mode='a') as f:
            with redirect_stdout(f):
                #print('run number index    total good asgn    total bad asgn   total edge asgn   total used asgn')
                print(n,'                     ',ngood_tot,'       ',nbad_tot,'      ',nedge_tot,'     ',nused_tot,'     ',miss)
    ###############################################################################
    #end of n-th ASAP run
    ###############################################################################
    with open(asgn_filename[n], mode='w') as f:
        with redirect_stdout(f):
            #print('istep is ',istep)
            print('istep is ',istep,'score is ',score)

            print('number of good asgn is ',ngood_tot,',number of biased good asgn is ',ngood_tot_bias,',number of bad asgn is ',nbad_tot,', number of edge asgn is ',nedge_tot,', number of used signals is ',nused_tot)


            for i in range(0,nres+1):
                print('residue ',i,'was chosen ',accept_ct[i],'times.')
            print('##########################################################')

            for i in range(0, npeak_nca+1):
                print('row ',i,'in ncacx input was picked ',nca_ct[i],'times.')
            print('##########################################################')
            #for i in range(0,nres+1):
                #print(current_nca[i],end='')
                #temp_id = current_nca[i]
                #print(' ',n15freq_nca[temp_id],csfreq_nca[temp_id][0],cfreq_nca[temp_id],end='')
                #for i1 in range(1,nfreq_nca-2):
                   # print(' ',csfreq_nca[temp_id][i1],end='')
                #print(' ',n15lw_nca[temp_id],cslw_nca[temp_id][0],clw_nca[temp_id],end='')
                #for i1 in range(1,nfreq_nca-2):
                    #print(' ',cslw_nca[temp_id][i1],end='')

                    #print(' ',deg_nca[temp_id],rtyp_nca[temp_id],end='')

                    #temp_id2 = current_nco[i]
                    #if temp_id2 != 0 and temp_id != 0:
                        #print(' ','matching nco signals are',end='' )
                        #temp_list = possmatch[temp_id][temp_id2]
                        #temp_len = len(temp_list)
                        #for j in range(temp_len):
                            #ksite = temp_list[j]
                            #if j<temp_len-1:
                                #print(' ',n15freq_nco[ksite],csfreq_nco[ksite],cfreq_nco[ksite],n15lw_nco[ksite],cslw_nco[ksite],clw_nco[ksite],deg_nco[ksite],rtyp_nco[ksite],end='')
                            #else:
                                #print(' ',n15freq_nco[ksite],csfreq_nco[ksite],cfreq_nco[ksite],n15lw_nco[ksite],cslw_nco[ksite],clw_nco[ksite],deg_nco[ksite],rtyp_nco[ksite])
                    #else:
                        #print()
    ####################################################################################
    #with open(ncacx4bk_filename[n], mode='w') as f:
        #with redirect_stdout(f):
    #record the full nca and nco asgn that can be retrieved for terminator adjustment retrospectively
    with open(ncacx4bk_filename[n], mode='w') as f:
        with redirect_stdout(f):
            print(npeak_nca,nfreqmax_nca)
            for i in range(1,nres+1):
                k_nca = current_nca[i]
                k_nco = current_nco[i]
                if k_nca != 0:
                    print(n15freq_nca[k_nca],csfreq_nca[k_nca][0],cfreq_nca[k_nca],end='')
                    for i1 in range(1,nfreqmax_nca-2):
                        print(' ',csfreq_nca[k_nca][i1],end='')
                    print(' ',n15lw_nca[k_nca],cslw_nca[k_nca][0],clw_nca[k_nca],end='')
                    for i1 in range(1,nfreqmax_nca-2):
                        print(' ',cslw_nca[k_nca][i1],end='')
                    #if consistency[ksite] == 1:
                        #print(' ',deg_nca[i],temp_rtyp)

                    if rigor > 0:
                        if overlap[k_nca] < 1 :
                            #temp_rtyp = extract_letter(rtyp_nca[i]) + str(ksite) #current location asgned to this nca
                            temp_rtyp = protein[i] + str(i)
                            ####################################################
                            if current_nco[i] > 0 :# to exclude the case if nco asgn at this site is zero(an edge asgn, because although this nca is consistently asgned, the nco pairing with it to be asgned is null)
                                klen = possmatch[k_nca][current_nco[i]]

                                for j in range(len(klen)):
                                    temp_id=klen[j]
                                    rtyp_nco[temp_id] = temp_rtyp
                            else:
                                temp_rtyp = extract_letter(rtyp_nca[k_nca])
                        else:
                            temp_rtyp = extract_letter(rtyp_nca[k_nca])
                    else:
                        #temp_rtyp = extract_letter(rtyp_nca[i]) + str(ksite) #current location asgned to this nca
                        if final == 1:
                            temp_rtyp = protein[i] + str(i)
                            ####################################################
                            if k_nco > 0 :# to exclude the case if nco asgn at this site is zero(an edge asgn, because although this nca is consistently asgned, the nco pairing with it to be asgned is null)
                                klen = possmatch[k_nca][k_nco]

                                for j in range(len(klen)):
                                    temp_id=klen[j]
                                    rtyp_nco[temp_id] = temp_rtyp
                        elif final == 0:
                            if k_nco >0:
                                temp_rtyp = protein[i] + str(i)
                                ####################################################
                                if k_nco > 0 :# to exclude the case if nco asgn at this site is zero(an edge asgn, because although this nca is consistently asgned, the nco pairing with it to be asgned is null)
                                    klen = possmatch[k_nca][k_nco]

                                    for j in range(len(klen)):
                                        temp_id=klen[j]
                                        rtyp_nco[temp_id] = temp_rtyp

                            elif k_nco == 0:
                                temp_rtyp = extract_letter(rtyp_nca[k_nca])

                    #else:
                        #temp_rtyp = extract_letter(rtyp_nca[k_nca])
                    #if current_nco[ksite] > 0 :# to exclude the case if nco asgn at this site is zero(an edge asgn, because although this nca is consistently asgned, the nco pairing with it to be asgned is null)
                    print(' ',1,temp_rtyp)
                    #mark its matched nco resonances to be printed next
            for i in range(1,npeak_nca+1):
                ksite = nca_asgn[i]
                if ksite == 0:
                   print(n15freq_nca[i],csfreq_nca[i][0],cfreq_nca[i],end='')
                   for i1 in range(1,nfreqmax_nca-2):
                       print(' ',csfreq_nca[i][i1],end='')
                   print(' ',n15lw_nca[i],cslw_nca[i][0],clw_nca[i],end='')
                   for i1 in range(1,nfreqmax_nca-2):
                       print(' ',cslw_nca[i][i1],end='')
                   print(' ',deg_nca[i]-used_nca[i],rtyp4bk_nca[i])
                elif ksite != 0 and deg_nca[i]-used_nca[i] > 0:
                    print(n15freq_nca[i],csfreq_nca[i][0],cfreq_nca[i],end='')
                    for i1 in range(1,nfreqmax_nca-2):
                        print(' ',csfreq_nca[i][i1],end='')
                    print(' ',n15lw_nca[i],cslw_nca[i][0],clw_nca[i],end='')
                    for i1 in range(1,nfreqmax_nca-2):
                        print(' ',cslw_nca[i][i1],end='')
                    print(' ',deg_nca[i]-used_nca[i],rtyp4bk_nca[i])





    with open(ncocx4bk_filename[n], mode='w') as f:
        with redirect_stdout(f):
            print(nsig_nco,3)
            for i in range(1,nsig_nco+1):
                print(n15freq_nco[i],csfreq_nco[i],cfreq_nco[i],n15lw_nco[i],cslw_nco[i],clw_nco[i],deg_nco[i],rtyp_nco[i])

    #####################################################################################

    #collect consistency informatio from each asgn run
    if n == 0:
        miss = 0
        reference_asgn = [0 for i in range(nres+1)]
        for i in range(1,nres+1):
            reference_asgn[i] = current_nca[i]
            if current_nca[i] == 0 :
                consistency[i] = 0
                reference_asgn[i] = 0

            elif current_nco[i] == 0:
                consistency[i] = 0
                reference_asgn[i] = 0
                if nmatch[current_nca[i]] > 0:# count how many of those with nco match not but its nco not asgned
                    if overlap[current_nca[i]] < 1 :
                          miss += 1

            if rigor == 1: #only asgn without entanglment will be counted towards consistently asgned to seed the next round
                if fix[current_nca[i]][current_nco[i]] < 0:#only those without entanglement of matched nco will seed next rd
                    consistency[i] = 0
                    reference_asgn[i] = 0

            if rigor == 2: #only nmatch = 1 will be counted towards consistently asgned to seed the next round
                if nmatch[current_nca[i]] > 1:
                    consistency[i] = 0
                    reference_asgn[i] = 0
                    if overlap[current_nca[i]] < 1 :
                        miss += 1


            if rigor == 3:#when rigor > 0, only those marked by non overlap count towards consistenly asgned.
                if overlap[current_nca[i]] > 0:
                    consistency[i] = 0
                    reference_asgn[i] = 0


            #mark its matched nco resonances to be printed next
            ##################################################
            #this section is to make sure the nco asgn paired with this consistenly asgned nca asgn reached global minimum
            #, if it is not accessed by mc attemp over the later half of the mcsa steps.

            #this section is to make sure the nco asgn paired with this consistenly asgned nca asgn reached global minimum
            #, if it is not accessed by mc attemp over the later half of the mcsa steps.
            flag = 0
            temp_step = terminator
            for k in range(temp_step,nstep+1):
                if occupancy_sum[k][i] != 0:
                    flag = 1
            ####################################################
            if flag == 1:
                #reference_asgn[i] = 0
                consistency[i] = 0

    elif n > 0:
        for i in range(1,nres+1):
            if reference_asgn[i] != 0:
                #if overlap[current_nca[i]] > 0:#it doesnt matter if they have zero overlap.

                if reference_asgn[i] != current_nca[i]:
                    consistency[i] = 0
                    reference_asgn[i] = 0
                if current_nco[i] == 0:
                    consistency[i] = 0
                    reference_asgn[i] = 0
                    if nmatch[current_nca[i]] > 0:
                        if overlap[current_nca[i]] < 1 :
                              miss += 1
            ##################################################
            #this section is to make sure the nco asgn paired with this consistenly asgned nca asgn reached global minimum
            #, if it is not accessed by mc attemp over the later half of the mcsa steps.

            #this section is to make sure the nco asgn paired with this consistenly asgned nca asgn reached global minimum
            #, if it is not accessed by mc attemp over the later half of the mcsa steps.
            flag = 0
            temp_step = terminator
            for k in range(temp_step,nstep+1):
                if occupancy_sum[k][i] != 0:
                    flag = 1
            ####################################################
            if flag == 1:
                #reference_asgn[i] = 0
                consistency[i] = 0
#########    ####################################################################################
for i in range(1,nres+1):
    if consistency[i] == 1:
        correct_asgn += 1
        temp_nca = current_nca[i]
        consistency_nca[temp_nca]=1
        #if priority[temp_nca] == 1:
        #rtyp_nca[temp_nca] = extract_letter(rtyp_nca[temp_nca])+str(i)
        #elif priority[temp_nca] == 2:
            #rtyp_nca[temp_nca] = rtyp_nca[temp_nca]
        #elif priority[temp_nca] == 0 :
            #rtyp_nca[temp_nca] = extract_letter(rtyp_nca[temp_nca])+' '+str(i)
        #rtyp_nca[temp_nca] = (rtyp_nca[temp_nca])+str(i)
        #temp_nco = current_nco[i]
        #temp_list = possmatch[temp_nca][temp_nco]
        #temp_len = len(temp_list)
        #for i1 in range(temp_len):
            #ksite = temp_list[i1]
            #rtyp_nco[ksite] = rtyp_nca[temp_nca]
    #else:
       # temp_nca = current_nca[i]
        #rtyp_nca[temp_nca] = extract_letter(rtyp4bk_nca[temp_nca])
        #temp_nco = current_nco[i]
        #if temp_nco != 0 and nmatch[temp_nca] > 0:

            #temp_list = possmatch[temp_nca][temp_nco]
            #temp_len = len(temp_list)
            #for i1 in range(temp_len):
                #ksite = temp_list[i1]
                #rtyp_nco[ksite] = extract_letter(rtyp4bk_nca[temp_nca])

with open(consistency_filename, mode='w') as f:
    with redirect_stdout(f):
        for i in range(nres+1):
            print(consistency[i])

with open(consistencynca_filename, mode='w') as f:
    with redirect_stdout(f):
        for i in range(npeak_nca+1):
            print(consistency_nca[i])
######################################################################################
print('Total of ',run_num,' Monte Carlo runs completes, and it took ', elapsed_time,' seconds.')
print()
print('There are ',correct_asgn,' residues consistently assigned in all MCSA runs.')

end_time = time.time()
elapsed_time = end_time - start_time
with open(runrecord_filename, mode='a') as f:
    with redirect_stdout(f):
        print('Total of ',run_num,' Monte Carlo runs completes, and it took ', elapsed_time,' seconds.')
        print()
        print('There are ',correct_asgn,' residues consistently assigned in all MCSA runs.')
##################################################################################
#record diagnostic information
if diagnostics == 1:
    #record the neighbor info of each nca asgn
    with open(neighbor_filename, mode='w') as f:
        with redirect_stdout(f):
            for i1 in range(nstep+1):
                for i2 in range(npeak_nca+1):
                    for i3 in range(npeak_nca+1):
                        neighbor_t[i1][i2][i3] /= run_num #normaliz by the total run num
                        #instigator[istep][i1] += neighbor[i1][i2][i3]
                        if i3 == 0:
                            print(neighbor_t[i1][i2][i3],end='')
                        elif i3 < npeak_nca:
                            print(' ',neighbor_t[i1][i2][i3],end='')
                        elif i3 == npeak_nca:
                            print(' ',neighbor_t[i1][i2][i3])
                            #print()

    ##################################################################################
    #record instigator of each step for the system
    with open(instigator_filename, mode='w') as f:
        with redirect_stdout(f):
            for i1 in range(nstep+1):
                for i2 in range(npeak_nca+1):
                    if i2 == 0:
                        print(instigator[i1][i2]/run_num,end='') # normalized to run_num
                    elif i2 <npeak_nca:
                        print(' ',instigator[i1][i2]/run_num,end='') # normalized to run_num
                    elif i2 == npeak_nca:
                        print(' ',instigator[i1][i2]/run_num) # normalized to run_num
    ######################################################################################
    #record at each MCSA step which nca asgn were allocated to each residue position for the system
    with open(occupancystep_filename, mode='w') as f:
        with redirect_stdout(f):
            for i1 in range(nstep+1):
                for i2 in range(nres+1):
                    for i3 in range(npeak_nca+1):

                        if i3 == 0:
                            print(occupancy_step[i1][i2][i3]/run_num,end='') #normaliz by the total run num
                        elif i3 < npeak_nca:
                            print(' ',occupancy_step[i1][i2][i3]/run_num,end='') #normaliz by the total run num
                        elif i3 == npeak_nca:
                            print(' ',occupancy_step[i1][i2][i3]/run_num) #normaliz by the total run num
                            #print()

    ##################################################################################
    #record which nca asgn were allocated to each residue position during the entire MCSA for the system
    with open(occupancysum_filename, mode='w') as f:
        with redirect_stdout(f):
            for i1 in range(nstep+1):
                for i2 in range(nres+1):
                    if i2 == 0:
                        print(occupancy_sum[i1][i2]/run_num,end='') # normalized to run_num
                    elif i2 <nres:
                        print(' ',occupancy_sum[i1][i2]/run_num,end='') # normalized to run_num
                    elif i2 == nres:
                        print(' ',occupancy_sum[i1][i2]/run_num) # normalized to run_num
    ####################################################################################

with open(knmatch_filename, mode='w') as f:
    with redirect_stdout(f):
        for i in range(npeak_nca+1):
            print(nmatch[i])

for i in range(1,npeak_nca+1):
    used_nca[i] = 0

with open(ncacx4nextrd_filename, mode='w') as f:
    with redirect_stdout(f):
        print(npeak_nca,nfreqmax_nca)
        for i in range(1,nres+1):
            if consistency[i] == 1:
                k_nca = current_nca[i]
                k_nco = current_nco[i]
                print(n15freq_nca[k_nca],csfreq_nca[k_nca][0],cfreq_nca[k_nca],end='')
                for i1 in range(1,nfreqmax_nca-2):
                    print(' ',csfreq_nca[k_nca][i1],end='')
                print(' ',n15lw_nca[k_nca],cslw_nca[k_nca][0],clw_nca[k_nca],end='')
                for i1 in range(1,nfreqmax_nca-2):
                    print(' ',cslw_nca[k_nca][i1],end='')
                if k_nca != 0 and k_nco != 0:
                    print(' ',1,protein[i] + str(i))
                    used_nca[k_nca] += 1
                    #mark its matched nco resonances to be printed next
                else:
                    print(' ',1,extract_letter(rtyp_nca[k_nca]))
                    used_nca[k_nca] += 1

        for i in range(1,npeak_nca+1):
            ksite = nca_asgn[i]
            if consistency[ksite] == 0:
               print(n15freq_nca[i],csfreq_nca[i][0],cfreq_nca[i],end='')
               for i1 in range(1,nfreqmax_nca-2):
                   print(' ',csfreq_nca[i][i1],end='')
               print(' ',n15lw_nca[i],cslw_nca[i][0],clw_nca[i],end='')
               for i1 in range(1,nfreqmax_nca-2):
                   print(' ',cslw_nca[i][i1],end='')
               print(' ',deg_nca[i]-used_nca[i],extract_letter(rtyp_nca[i]))





with open(ncocx4nextrd_filename, mode='w') as f:
    with redirect_stdout(f):
        print(nsig_nco,3)
        for i in range(1,nsig_nco+1):
            print(n15freq_nco[i],csfreq_nco[i],cfreq_nco[i],n15lw_nco[i],cslw_nco[i],clw_nco[i],deg_nco[i],rtyp_nco[i])

#########################################################################################
#record the num of good, bad, edge and tot used along MCSA
with open(numberrecord_filename, mode='w') as f:
    with redirect_stdout(f):
        for i in range(1,nstep+1):
            print(all_number[i][0]/run_num,all_number[i][1]/run_num,all_number[i][2]/run_num,all_number[i][3]/run_num)

############################################################################################

#########################################################################################
detector = [0 for i in range(npeak_nca+1)]
for i in range(1,npeak_nca+1):
    k_home = nca_asgn[i]
    if k_home != 0:
        ksite = current_nca[k_home]
        if ksite == i:
            detector[i] = 1
    else:
        detector[i] = 1

with open(detector_filename, mode='w') as f:
    with redirect_stdout(f):
        for i in range(npeak_nca+1):
            print(detector[i])



