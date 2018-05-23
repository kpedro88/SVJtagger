from ROOT import TTree, TFile, gROOT, gStyle, TH1F, TCanvas, TGraph, TLegend
from officialStyle import officialStyle
import numpy
gROOT.SetBatch(True)
#gROOT.SetBatch(False)
officialStyle(gStyle)
gStyle.SetOptTitle(0)


def rocCurve(hS, hB, label):
  ''' Create a ROC TGraph from two input histograms.
  '''
  maxBin = hS.GetNbinsX()

  if hS.Integral() == 0.:
    print 'ROC curve creator, hist', hS.GetName(), 'has zero entries'
    return

  if label == 'right':
    effsS = [hS.Integral(nBin, maxBin+1)/hS.Integral(0, maxBin+1) for nBin in range(0, maxBin + 1) ]
    rejB = [hB.Integral(nBin, maxBin+1)/hB.Integral(0, maxBin+1) for nBin in range(0, maxBin + 1) ]

  elif label == 'left':
    effsS = [hS.Integral(0, nBin)/hS.Integral(0, maxBin+1) for nBin in range(0, maxBin + 1) ]
    rejB = [hB.Integral(0, nBin)/hB.Integral(0, maxBin+1) for nBin in range(0, maxBin + 1) ]

  rocCurve = TGraph(maxBin, numpy.asarray(effsS), numpy.asarray(rejB))

  return rocCurve



def LegendSettings(leg):
    leg.SetBorderSize(0)
    leg.SetTextSize(0.032)
    leg.SetLineColor(0)
    leg.SetLineStyle(1)
    leg.SetLineWidth(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)



vardir = {
    'axis2':{'drawname':'axis2', 'nbins':1000, 'min':0, 'max':8, 'label':'-log(axis2)', 'type':'right'},
    'ptD':{'drawname':'ptD', 'nbins':1000, 'min':0, 'max':1, 'label':'ptD','type':'right'},
    'axis1':{'drawname':'axis1', 'nbins':1000, 'min':0, 'max':8, 'label':'-log(axis1)','type':'right'},
    'pt_dr_log':{'drawname':'pt_dr_log/pt', 'nbins':1000, 'min':0, 'max':1.5, 'label':'sum(log(pT/dR))','type':'left'},
    'charged_multiplicity':{'drawname':'charged_multiplicity', 'nbins':100, 'min':0, 'max':100, 'label':'charged mult.','type':'left'},
    'neutral_multiplicity':{'drawname':'neutral_multiplicity', 'nbins':100, 'min':0, 'max':100, 'label':'neutral mult.','type':'left'},

    }



file = TFile("combined_v2.root")
tree = file.Get("tree")

#sel_quark = "partonId!=21 && axis2 < 8 && jetIdLevel==3 && matchedJet==1 && nGenJetsInCone==1 && nGenJetsForGenParticle==1 && nJetsForGenParticle==1 && partonId < 4 && balanced==1 && mult >= 3 && mult <= 143";
#sel_gluon = "partonId==21 && axis2 < 8 && jetIdLevel==3 && matchedJet==1 && nGenJetsInCone==1 && nGenJetsForGenParticle==1 && nJetsForGenParticle==1 && balanced==1 && mult >= 3 && mult <= 143";

sel_quark = "abs(eta) > 3 && pt > 63.2 && pt < 79.6 && partonId!=21";
sel_gluon = "abs(eta) > 3 && pt > 63.2 && pt < 79.6 && partonId==21";


graphs = []

leg = TLegend(0.2,0.55,0.5,0.9)
LegendSettings(leg)

for varname, var in vardir.iteritems():        
    hname_q = 'hist_quark_' + varname
    hname_g = 'hist_gluon_' + varname

    hist_q = TH1F(hname_q, hname_q, var['nbins'], var['min'], var['max'])
    hist_g = TH1F(hname_g, hname_g, var['nbins'], var['min'], var['max'])

    tree.Draw(var['drawname'] + ' >> ' + hname_q, sel_quark)
    tree.Draw(var['drawname'] + ' >> ' + hname_g, sel_gluon)

    graph = rocCurve(hist_q, hist_g, var['type'])

    graph.GetXaxis().SetTitle('Signal eff.')
    graph.GetYaxis().SetTitle('Bkg. eff.')
    graph.SetLineWidth(3)
    graph.SetMarkerSize(0.)
    graphs.append(graph)

    hname = 'hist_' + varname
    hist = TH1F(hname, hname, 1000,0,1)
  
    for ibin in range(1, hist.GetXaxis().GetNbins()+1):
      hist.SetBinContent(ibin, graph.Eval(hist.GetXaxis().GetBinCenter(ibin)))
    
#    graphs.append(graph)   

    fraction = hist.Integral()*0.001


    leg.AddEntry(graph, var['label'] + ', ROC = {0:.3f}'.format(fraction), 'lep')

canvas = TCanvas('can')
canvas.SetGridx()
canvas.SetGridy()

for gindex, graph in enumerate(graphs):
  graph.SetLineColor(gindex+1)
#  graph.SetLineStyle(gindex+1)
  graph.GetXaxis().SetRangeUser(0,1)
  graph.GetYaxis().SetRangeUser(0,1)
  graph.GetYaxis().SetNdivisions(508)

  if gindex==0:
    graph.Draw("apl")
  else:
    graph.Draw("plsame")

leg.Draw()
canvas.Print('plots/compare_roc.gif')


