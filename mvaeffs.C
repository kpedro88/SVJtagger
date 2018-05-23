
#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;

#include "tmvaglob.C"

#include "RQ_OBJECT.h"

#include "TH1.h"
#include "TROOT.h"
#include "TList.h"
#include "TIterator.h"
#include "TStyle.h"
#include "TPad.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLine.h"
#include "TH2.h"
#include "TFormula.h"
#include "TFile.h"
#include "TApplication.h"
#include "TKey.h"
#include "TClass.h"

#include "TGWindow.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"

// this macro plots the signal and background efficiencies
// as a function of the MVA cut.

enum PlotType { EffPurity = 0 };

class MethodInfo : public TNamed {
public:
   MethodInfo() :
      methodName(""),
      methodTitle(""),
      sig(0),
      bgd(0),
      origSigE(0),
      origBgdE(0),
      sigE(0),
      bgdE(0),
      purS(0),
      sSig(0),
      effpurS(0),
      canvas(0),
      line1(0),
      line2(0),
      fOne(0)
   {}
   virtual ~MethodInfo();

   TString  methodName;
   TString  methodTitle;
   TH1*     sig;
   TH1*     bgd;
   TH1*     origSigE;
   TH1*     origBgdE;
   TH1*     sigE;
   TH1*     bgdE;
   TH1*     purS;
   TH1*     sSig;    
   TH1*     effpurS;
   TCanvas* canvas;
   TLatex*  line1;
   TLatex*  line2;
   TLine*   fOne;

   void SetResultHists() 
   {
      TString pname    = "purS_"         + methodTitle;
      TString epname   = "effpurS_"      + methodTitle;
      TString ssigname = "significance_" + methodTitle;

      sigE = (TH1*)origSigE->Clone("sigEffi");
      bgdE = (TH1*)origBgdE->Clone("bgdEffi");
      
      Int_t nbins = sigE->GetNbinsX();
      Double_t low = sigE->GetBinLowEdge(1);
      Double_t high = sigE->GetBinLowEdge(nbins+1);
      purS    = new TH1F(pname, pname, nbins, low, high);
      sSig    = new TH1F(ssigname, ssigname, nbins, low, high);
      effpurS = new TH1F(epname, epname, nbins, low, high);        

      // chop off useless stuff
      sigE->SetTitle( Form("Cut efficiencies for %s classifier", methodTitle.Data()) );
         
      // set the histogram style
      TMVAGlob::SetSignalAndBackgroundStyle( sigE, bgdE );
      TMVAGlob::SetSignalAndBackgroundStyle( purS, bgdE );
      TMVAGlob::SetSignalAndBackgroundStyle( effpurS, bgdE );
      sigE->SetFillStyle( 0 );
      bgdE->SetFillStyle( 0 );
      sSig->SetFillStyle( 0 );
      sigE->SetLineWidth( 3 );
      bgdE->SetLineWidth( 3 );
      sSig->SetLineWidth( 3 );
      sSig->SetLineColor( 6 );
      sSig->GetYaxis()->SetAxisColor(6);

      // the purity and quality
      purS->SetFillStyle( 0 );
      purS->SetLineWidth( 2 );
      purS->SetLineStyle( 5 );
      effpurS->SetFillStyle( 0 );
      effpurS->SetLineWidth( 2 );
      effpurS->SetLineStyle( 6 );
   }

   ClassDef(MethodInfo,0)
};

MethodInfo::~MethodInfo() 
{
   delete sigE;
   delete bgdE;
   delete purS;
   delete sSig;
   delete effpurS;
   delete canvas;
}

class StatDialogMVAEffs {  

   RQ_OBJECT("StatDialogMVAEffs")

 public:

   StatDialogMVAEffs(const TGWindow* p, Float_t ns, Float_t nb);
   virtual ~StatDialogMVAEffs() {}

   void SetFormula(const TString& f) { fFormula = f; }
   TString GetFormula();
   TString GetLatexFormula();
   
   void ReadHistograms(TFile* file);
   void UpdateSignificanceHists();
   void DrawHistograms();

 private:

   TGMainFrame *fMain;
   Float_t fNSignal;
   Float_t fNBackground;  
   TString fFormula;
   TList * fInfoList;

   TGNumberEntry* fSigInput;
   TGNumberEntry* fBkgInput;

   TGHorizontalFrame* fButtons;
   TGTextButton* fOkBut;
   TGTextButton* fCnclBut;

   Int_t maxLenTitle;

   void UpdateCanvases();

   // slots
   void SetNSignal(); // *SIGNAL*
   void SetNBackground(); // *SIGNAL*
   void Redraw(); // *SIGNAL*
   void Close(); // *SIGNAL*

   // result printing
   void PrintResults( const MethodInfo* info );
};

void StatDialogMVAEffs::SetNSignal() 
{
   fNSignal = fSigInput->GetNumber();
}

void StatDialogMVAEffs::SetNBackground() 
{
   fNBackground = fBkgInput->GetNumber();
}

TString StatDialogMVAEffs::GetFormula() 
{
   TString f = fFormula;
   f.ReplaceAll("S","x");
   f.ReplaceAll("B","y");
   return f;
}

TString StatDialogMVAEffs::GetLatexFormula() 
{
   TString f = fFormula;
   f.ReplaceAll("(","{");
   f.ReplaceAll("}","}");
   f.ReplaceAll("sqrt","#sqrt");
   return f;
}

void StatDialogMVAEffs::Redraw() 
{
   UpdateSignificanceHists();
   UpdateCanvases();
}

void StatDialogMVAEffs::Close() 
{
   if (fInfoList) { 
      TIter next(fInfoList);
      MethodInfo *info(0);
      while ( (info = (MethodInfo*)next()) ) {
         delete info;
      }
      delete fInfoList;
      fInfoList=0;
   }
   fMain->CloseWindow();
   fMain->Cleanup();
   fMain = 0;

   delete this;
}

StatDialogMVAEffs::StatDialogMVAEffs(const TGWindow* p, Float_t ns, Float_t nb) :
   fNSignal(ns),
   fNBackground(nb),
   fFormula(""),
   fInfoList(0),
   fSigInput(0),
   fBkgInput(0),
   fButtons(0),
   fOkBut(0),
   fCnclBut(0),
   maxLenTitle(0)
{
   UInt_t totalWidth  = 500;
   UInt_t totalHeight = 300;

   // main frame
   fMain = new TGMainFrame(p, totalWidth, totalHeight, kMainFrame | kVerticalFrame);

   TGLabel *sigLab = new TGLabel(fMain,"Signal events");
   fMain->AddFrame(sigLab, new TGLayoutHints(kLHintsLeft | kLHintsTop,5,5,5,5));

   fSigInput = new TGNumberEntry(fMain, (Double_t) fNSignal,5,-1,(TGNumberFormat::EStyle) 5);
   fMain->AddFrame(fSigInput, new TGLayoutHints(kLHintsLeft | kLHintsTop,5,5,5,5));
   fSigInput->Resize(100,24);

   TGLabel *bkgLab = new TGLabel(fMain, "Background events");
   fMain->AddFrame(bkgLab, new TGLayoutHints(kLHintsLeft | kLHintsTop,5,5,5,5));

   fBkgInput = new TGNumberEntry(fMain, (Double_t) fNBackground,5,-1,(TGNumberFormat::EStyle) 5);
   fMain->AddFrame(fBkgInput, new TGLayoutHints(kLHintsLeft | kLHintsTop,5,5,5,5));
   fBkgInput->Resize(100,24);

   fButtons = new TGHorizontalFrame(fMain, totalWidth,30);

   fCnclBut = new TGTextButton(fButtons,"&Close");
   fCnclBut->Connect("Clicked()", "StatDialogMVAEffs", this, "Close()");
   fButtons->AddFrame(fCnclBut, new TGLayoutHints(kLHintsLeft | kLHintsTop));

   fOkBut = new TGTextButton(fButtons,"&Draw");
   fButtons->AddFrame(fOkBut, new TGLayoutHints(kLHintsRight | kLHintsTop,15));
  
   fMain->AddFrame(fButtons,new TGLayoutHints(kLHintsLeft | kLHintsBottom,5,5,5,5));

   fMain->SetWindowName("Significance");
   fMain->MapSubwindows();
   fMain->Resize(fMain->GetDefaultSize());
   fMain->MapWindow();

   fSigInput->Connect("ValueSet(Long_t)","StatDialogMVAEffs",this, "SetNSignal()");
   fBkgInput->Connect("ValueSet(Long_t)","StatDialogMVAEffs",this, "SetNBackground()");

   fOkBut->Connect("Clicked()","TGNumberEntry",fSigInput, "ValueSet(Long_t)");
   fOkBut->Connect("Clicked()","TGNumberEntry",fBkgInput, "ValueSet(Long_t)");
   fOkBut->Connect("Clicked()", "StatDialogMVAEffs", this, "Redraw()");   
}

void StatDialogMVAEffs::UpdateCanvases() 
{
   if (fInfoList==0) return;
   if (fInfoList->First()==0) return;
   MethodInfo* info = (MethodInfo*)fInfoList->First();
   if ( info->canvas==0 ) {
      DrawHistograms();
      return;
   }
   TIter next(fInfoList);
   while ( (info = (MethodInfo*)next()) ) {
      info->canvas->Modified(kTRUE);
      info->canvas->Update();
      info->canvas->Paint();
   }
}

void StatDialogMVAEffs::UpdateSignificanceHists() 
{
   TFormula f("sigf",GetFormula());
   TIter next(fInfoList);
   MethodInfo * info(0);
   TString cname = "Classifier";
   if (cname.Length() >  maxLenTitle)  maxLenTitle = cname.Length();
   TString str = Form( "%*s   (   #signal, #backgr.)  Optimal-cut  S/sqrt(S+B)  Eff-sig  Eff-bkg", 
                       maxLenTitle, cname.Data() );
   cout << "--- " << setfill('=') << setw(str.Length()) << "" << setfill(' ') << endl;
   cout << "--- " << str << endl;
   cout << "--- " << setfill('-') << setw(str.Length()) << "" << setfill(' ') << endl;
   while( (info = (MethodInfo*) next()) ) {
      for (Int_t i=1; i<=info->origSigE->GetNbinsX(); i++) {
         Float_t eS = info->origSigE->GetBinContent( i ) * fNSignal;
         Float_t eB = info->origBgdE->GetBinContent( i ) * fNBackground;
         info->purS->SetBinContent( i, eS/(eS+eB) );
         info->sSig->SetBinContent( i, f.Eval(eS,eB) );
         info->effpurS->SetBinContent( i, eS*info->purS->GetBinContent( i ) );
      }
      
      Float_t maxSig = info->sSig->GetMaximum();

      // scale efficiency histograms to 80% of the max of the significance
      info->sigE->Scale(.8*maxSig/info->sigE->GetMaximum());
      info->bgdE->Scale(.8*maxSig/info->bgdE->GetMaximum());
      info->purS->Scale(.8*maxSig/info->purS->GetMaximum());

      info->effpurS->Scale(maxSig/info->effpurS->GetMaximum());

      // update the text in the lower left corner
      PrintResults( info );
  
      if (info->fOne!=0) {
         Float_t one = info->sigE->GetMaximum();
         info->fOne->SetY1(one); info->fOne->SetY2(one);
      }

   }
}

void StatDialogMVAEffs::ReadHistograms(TFile* file) 
{
   fInfoList = new TList;

   // search for the right histograms in full list of keys
   TIter next(file->GetListOfKeys());
   TKey *key(0);   
   while( (key = (TKey*)next()) ) {

      if (!TString(key->GetName()).BeginsWith("Method_")) continue;
      if( ! gROOT->GetClass(key->GetClassName())->InheritsFrom("TDirectory") ) continue;

      cout << "--- Found directory: " << ((TDirectory*)key->ReadObj())->GetName() << endl;

      TDirectory* mDir = (TDirectory*)key->ReadObj();

      TIter keyIt(mDir->GetListOfKeys());
      TKey *titkey;
      while((titkey = (TKey*)keyIt())) {
        if( ! gROOT->GetClass(titkey->GetClassName())->InheritsFrom("TDirectory") ) continue;
        
        MethodInfo* info = new MethodInfo();
        TDirectory* titDir = (TDirectory *)titkey->ReadObj();

        TMVAGlob::GetMethodName(info->methodName,key);
        TMVAGlob::GetMethodTitle(info->methodTitle,titDir);        
        if (info->methodTitle.Length() > maxLenTitle) maxLenTitle = info->methodTitle.Length();
        TString hname = "MVA_" + info->methodTitle;
        
        cout << "--- Classifier: " << info->methodTitle << endl;
        
        info->sig = dynamic_cast<TH1*>(titDir->Get( hname + "_S" ));
        info->bgd = dynamic_cast<TH1*>(titDir->Get( hname + "_B" ));
        info->origSigE = dynamic_cast<TH1*>(titDir->Get( hname + "_effS" ));
        info->origBgdE = dynamic_cast<TH1*>(titDir->Get( hname + "_effB" ));      
        if (info->sig==0 || info->bgd==0) { delete info; continue; }

        info->SetResultHists();
        fInfoList->Add(info);
      }
   }
   return;
}

void StatDialogMVAEffs::DrawHistograms() 
{
   // counter variables
   Int_t countCanvas = 0;

   // define Canvas layout here!
   const Int_t width = 600;   // size of canvas

   TIter next(fInfoList);
   MethodInfo* info(0);
   while ( (info = (MethodInfo*)next()) ) {

      // create new canvas
      TCanvas *c = new TCanvas( Form("canvas%d", countCanvas+1), 
                                Form("Cut efficiencies for %s classifier",info->methodTitle.Data()), 
                                countCanvas*50+200, countCanvas*20, width, Int_t(width*0.78) ); 
      info->canvas = c;

      // draw grid
      c->SetGrid(1);
         
      TStyle *TMVAStyle = gROOT->GetStyle("Plain"); // our style is based on Plain
      TMVAStyle->SetLineStyleString( 5, "[32 22]" );
      TMVAStyle->SetLineStyleString( 6, "[12 22]" );
         
      // frame limits (choose judicuous x range)
      Float_t xmin = info->sig->GetXaxis()->GetXmin();
      Float_t xmax = info->sig->GetXaxis()->GetXmax();
             
      c->SetTopMargin(.2);
      
      // and the signal purity and quality
      info->sSig->SetTitle("Efficiency");
      info->sSig->Draw("");
      info->effpurS->Draw("samehistl");
      info->purS->Draw("samehistl");

      // overlay signal and background histograms
      info->sigE->Draw("samehistl");
      info->bgdE->Draw("samehistl");

      // add line to indicate eff=1
      Float_t one = info->sigE->GetMaximum();
      info->fOne = new TLine( xmin, one, xmax, one );
      info->fOne->SetLineWidth(1);
      info->fOne->SetLineStyle(1);
      info->fOne->Draw();
                    
      // Draw legend               
      TLegend *legend1= new TLegend( c->GetLeftMargin(), 1 - c->GetTopMargin(), 
                                     c->GetLeftMargin() + 0.4, 1 - c->GetTopMargin() + 0.12 );
      legend1->SetFillStyle( 1 );
      legend1->AddEntry(info->sigE,"Signal efficiency","L");
      legend1->AddEntry(info->bgdE,"Background efficiency","L");
      legend1->Draw("same");
      legend1->SetBorderSize(1);
      legend1->SetMargin( 0.3 );

      TLegend *legend2= new TLegend( c->GetLeftMargin() + 0.4, 1 - c->GetTopMargin(), 
                                     1 - c->GetRightMargin(), 1 - c->GetTopMargin() + 0.12 );
      legend2->SetFillStyle( 1 );
      legend2->AddEntry(info->purS,"Signal purity","L");
      legend2->AddEntry(info->effpurS,"Signal efficiency*purity","L");
      legend2->AddEntry(info->sSig,"S / #sqrt{S+B}","L");
      legend2->Draw("same");
      legend2->SetBorderSize(1);
      legend2->SetMargin( 0.3 );
         
      // print comments
      TLatex tl;
      tl.SetNDC();
      tl.SetTextSize( 0.033 );
      Int_t maxbin = info->sSig->GetMaximumBin();
      info->line1 = tl.DrawLatex( 0.15, 0.23, Form("For %1.0f signal and %1.0f background", fNSignal, fNBackground));
      tl.DrawLatex( 0.15, 0.19, "events the maximum S / #sqrt{S+B} is");
      info->line2 = tl.DrawLatex( 0.15, 0.15, Form("%3.4f when cutting at %3.4f",
                                             info->sSig->GetBinContent(maxbin), 
                                             info->sSig->GetXaxis()->GetBinCenter(maxbin)) );

      // save canvas to file
      c->Update();

      // switches
      const Bool_t Save_Images = kTRUE;

      if (Save_Images) {
         TMVAGlob::imgconv( c, Form("plots/mvaeffs_%s", info->methodTitle.Data()) ); 
      }
      countCanvas++;
   }
}

void StatDialogMVAEffs::PrintResults( const MethodInfo* info )
{
   Int_t maxbin = info->sSig->GetMaximumBin();
   if (info->line1 !=0 )
      info->line1->SetText( 0.15, 0.23, Form("For %1.0f signal and %1.0f background", fNSignal, fNBackground));
   
   if (info->line2 !=0 ) {
      Int_t maxbin = info->sSig->GetMaximumBin();
      info->line2->SetText( 0.15, 0.15, Form("%3.4f when cutting at %3.4f", info->sSig->GetMaximum(), 
                                             info->sSig->GetXaxis()->GetBinCenter(maxbin)) );
   }

   TString opt = Form( "%%%is:  (%%10.9g,%%9.8g)    %%9.4f   %%10.6g %%8.4g %%8.4g", maxLenTitle );
   cout << "--- " 
        << Form( opt.Data(),
                 info->methodTitle.Data(), fNSignal, fNBackground, 
                 info->sSig->GetXaxis()->GetBinCenter( maxbin ),
                 info->sSig->GetMaximum(), 
                 info->origSigE->GetBinContent( maxbin ),
                 info->origBgdE->GetBinContent( maxbin ) )
        << endl;
}

void mvaeffs( TString fin = "TMVA.root", PlotType ptype = EffPurity, 
              Bool_t useTMVAStyle = kTRUE, TString formula="S/sqrt(S+B)" )
{
   TMVAGlob::Initialize( useTMVAStyle );

   TFile* file = TMVAGlob::OpenFile( fin );  

   StatDialogMVAEffs* gGui = new StatDialogMVAEffs(gClient->GetRoot(), 1000, 1000);

   gGui->SetFormula(formula);

   gGui->ReadHistograms(file);

   gGui->UpdateSignificanceHists();

   gGui->DrawHistograms();
}
