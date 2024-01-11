import ROOT
import os
 
TMVA = ROOT.TMVA
TFile = ROOT.TFile
 
TMVA.Tools.Instance()

useBDT = True


inputFile = TFile.Open("period34partial.root")

# --- Register the training and test trees
protonTree = inputFile.Get("mytestbeamana/protonTree")
protonTreeNew = inputFile.Get("mytestbeamana/protonTreeNew")

pimuTree = inputFile.Get("mytestbeamana/pimuTree")
pimuTreeNew = inputFile.Get("mytestbeamana/pimuTreeNew")

kaonTree = inputFile.Get("mytestbeamana/kaonTree")
kaonTreeNew = inputFile.Get("mytestbeamana/kaonTreeNew")

ckovTree = inputFile.Get("mytestbeamana/ckovTree")

otherTree = inputFile.Get("mytestbeamana/otherTree")
otherTreeNew = inputFile.Get("mytestbeamana/otherTreeNew")

nomasscutTree = inputFile.Get("mytestbeamana/nomasscutTree")




sig = protonTree
back = pimucutTree


signame = (f"{sig}").split()[1]
backname = (f"{back}").split()[1]

outputFile = TFile.Open(f"{signame}VS{backname}.root", "RECREATE")

factory =TMVA.Factory( "TMVAClassificationCategory", outputFile)

loader = TMVA.DataLoader("dataset")
 
loader.AddVariable("hit_n")
loader.AddVariable("hit_averagex")
loader.AddVariable("hit_averagey")
loader.AddVariable("hit_averagepe")
loader.AddVariable("hit_firstx")
loader.AddVariable("hit_firsty")
loader.AddVariable("hit_firstpe")
loader.AddVariable("hit_width")
#loader.AddVariable("track_energy")
#loader.AddVariable("track_mintime")
#loader.AddVariable("track_nhit")
#loader.AddVariable("track_x")
#loader.AddVariable("track_y")
#loader.AddVariable("track_z")
#loader.AddVariable("wc_x")
#loader.AddVariable("wc_y")
loader.AddSpectator("tof_time")
loader.AddSpectator("wc_p")
loader.AddSpectator("wc_mass")




signalWeight = 1.0
backgroundWeight = 1.0
# You can add an arbitrary number of signal or background trees
loader.AddSignalTree(sig,signalWeight)
loader.AddBackgroundTree(back, backgroundWeight)

mycuts = ROOT.TCut("")  # for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";

loader.PrepareTrainingAndTestTree( mycuts, "SplitMode=random:!V" )

factory.BookMethod(loader,TMVA.Types.kBDT,"BDT","BoostType=Grad")
factory.BookMethod(loader, TMVA.Types.kCuts, "Cuts");

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
c1 = factory.GetROCCurve(loader)
c1.Draw()
outputFile.Close()
