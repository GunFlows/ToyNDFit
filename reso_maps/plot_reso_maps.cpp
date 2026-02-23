

void plot_reso_maps(){

  TFile *f_pmu = new TFile("pmu_resolution_map.root");
  TFile *f_thetamu = new TFile("costhetamu_resolution_map.root");

  // open canvas in file 1 and edit titles
  TCanvas *c1 = (TCanvas*)f_pmu->Get("pmu_resolution_map");
  // find histogram (loop over primitives in file)
  TH2D *h_pmu = nullptr;
  TIter next(f_pmu->GetListOfKeys());
  TKey *key;
  while ((key = (TKey*)next())) {
    if (strcmp(key->GetClassName(), "TH2D") == 0) {
      h_pmu = (TH2D *) key->ReadObj();
      break;
    }
  }
  if (h_pmu) {
    h_pmu->SetTitle("Muon Momentum Resolution;True P_{#mu} [GeV/c];True cos(#theta_{#mu})");
  } else {
    std::cerr << "Error: TH2D histogram not found in pmu_resolution_map.root" << std::endl;
  }

  // open canvas in file 2 and edit titles
  TCanvas *c2 = (TCanvas*)f_thetamu->Get("costhetamu_resolution_map");
  // find histogram (loop over primitives in file)
  TH2D *h_thetamu = nullptr;
  TIter next2(f_thetamu->GetListOfKeys());
  while ((key = (TKey*)next2())) {
    if (strcmp(key->GetClassName(), "TH2D") == 0) {
      h_thetamu = (TH2D *) key->ReadObj();
      break;
    }
  }
  if (h_thetamu) {
    h_thetamu->SetTitle("Muon Angle Resolution;True P_{#mu} [GeV/c];True cos(#theta_{#mu})");
  } else {
    std::cerr << "Error: TH2D histogram not found in costhetamu_resolution_map.root" << std::endl;
  }

  // draw in same canvas (side by side)
  TCanvas *c = new TCanvas("c", "Resolution Maps", 1600, 600);
  c->Divide(2,1);
  c->cd(1);
  gPad->SetLeftMargin(0.15);
  gPad->SetRightMargin(0.2);
  if (h_pmu) {
    h_pmu->Draw("COLZ");
    h_pmu->GetYaxis()->SetRangeUser(0.7,1);
    h_pmu->GetZaxis()->SetTitleOffset(1.5);
    h_pmu->GetXaxis()->SetTitleOffset(1.3);
  }
  gPad->SetLogx();
  c->cd(2);
  gPad->SetLeftMargin(0.15);
  gPad->SetRightMargin(0.2);
  if (h_thetamu) {
    h_thetamu->Draw("COLZ");
    h_thetamu->GetYaxis()->SetRangeUser(0.7,1);
    h_thetamu->GetZaxis()->SetTitleOffset(2.);
    h_thetamu->GetXaxis()->SetTitleOffset(1.3);
  }
  gPad->SetLogx();

  c->SaveAs("resolution_maps.pdf");




}