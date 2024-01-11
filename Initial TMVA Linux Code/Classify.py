import ROOT
import os
 
TMVA = ROOT.TMVA
TFile = ROOT.TFile
 
TMVA.Tools.Instance()

useBDT = True


inputFile = TFile.Open("trackana_p234_2004.root")

# --- Register the training and test trees
protonTree = inputFile.Get("testbeamtrackana/protonTree")

pimuTree = inputFile.Get("testbeamtrackana/pimuTree")

kaonTree = inputFile.Get("testbeamtrackana/kaonTree")

ckovTree = inputFile.Get("testbeamtrackana/ckovTree")

otherTree = inputFile.Get("testbeamtrackana/otherTree")

tracktree = inputFile.Get("testbeamtrackana/tracktree")

testbeamtrackana = inputFile.Get("testbeamtrackana")





sig = protonTree
back = pimuTree


signame = (f"{sig}").split()[1]
backname = (f"{back}").split()[1]

outputFile = TFile.Open(f"{signame}VS{backname}.root", "RECREATE")

factory =TMVA.Factory( "TMVAClassificationCategory", outputFile)

loader = TMVA.DataLoader("dataset")
 
loader.AddVariable("_hit_n")
loader.AddVariable("_hit_avgdr")
loader.AddVariable("_hit_avgpe")
loader.AddVariable("_hit_firstdr")
loader.AddVariable("_hit_firstpe")
loader.AddVariable("_hit_lastdr")
loader.AddVariable("_hit_lastdz")
loader.AddVariable("_hit_width")
loader.AddVariable("_chit_gev")
loader.AddVariable("_hit_totpe")
loader.AddVariable("_hit_firstdz")
loader.AddVariable("_hit_lastpe")
loader.AddVariable("_chit_frac")

#loader.AddVariable("track_energy")
#loader.AddVariable("track_mintime")
#loader.AddVariable("track_nhit")
#loader.AddVariable("track_x")
#loader.AddVariable("track_y")
#loader.AddVariable("track_z")
#loader.AddVariable("wc_x")
#loader.AddVariable("wc_y")
#loader.AddSpectator("tof_time")
#loader.AddSpectator("wc_p")
#loader.AddSpectator("wc_mass")




signalWeight = 1.0
backgroundWeight = 1.0
# You can add an arbitrary number of signal or background trees
loader.AddSignalTree(sig,signalWeight)
loader.AddBackgroundTree(back, backgroundWeight)

mycuts = ROOT.TCut("_pass_intimehit==1 && _pass_wcntrack==1")  # for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";

loader.PrepareTrainingAndTestTree( mycuts, "SplitMode=random:!V" )

factory.BookMethod(loader,TMVA.Types.kBDT,"BDT","BoostType=Grad")
factory.BookMethod(loader, TMVA.Types.kCuts, "Cuts");

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
c1 = factory.GetROCCurve(loader)
c1.Draw()
outputFile.Close()
