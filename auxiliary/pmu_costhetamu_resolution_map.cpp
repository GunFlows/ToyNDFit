

void pmu_costhetamu_resolution_map(const char * filename = "run9wMCsplines.root"){

  // Batch mode
  gROOT->SetBatch(kTRUE);
  // Quiet root
  gErrorIgnoreLevel = kError;

  // Open TFile
  TFile * f = TFile::Open(filename);
  if(!f || f->IsZombie()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  TTree * tree = (TTree*) f->Get("sample_sum");
  if(!tree) {
    std::cerr << "Error: TTree 'sample_sum' not found in file: " << filename << std::endl;
    f->Close();
    return;
  }

  // Set up variables to read from the tree
  double Pmu;
  double TruePmu;
  double CosThetamu;
  double TrueCosThetamu;

  tree->SetBranchAddress("Pmu", &Pmu);
  tree->SetBranchAddress("TruePmu", &TruePmu);
  tree->SetBranchAddress("CosThetamu", &CosThetamu);
  tree->SetBranchAddress("TrueCosThetamu", &TrueCosThetamu);

  vector<double> pmu_bins = {0,300,400,800,1500,5000,30000};
  vector<double> costhetamu_bins = {-1,0.7,0.8,0.9,0.95,0.99,1};

  map<pair<double,double>, TH1F*> pmu_resolution_hist_map;
  map<pair<double,double>, TH1F*> costhetamu_resolution_hist_map;


  for(double pmu : pmu_bins){
    for(double costhetamu : costhetamu_bins){
      TString pmu_hist_name = Form("pmu_resolution_Pmu_%.0f_CosThetamu_%.2f", pmu, costhetamu);
      TString costhetamu_hist_name = Form("costhetamu_resolution_Pmu_%.0f_CosThetamu_%.2f", pmu, costhetamu);

      TH1F* deltapmu_hist = new TH1F(pmu_hist_name, pmu_hist_name, 300, -1, 1);
      TH1F* deltacosthetamu_hist = new TH1F(costhetamu_hist_name, costhetamu_hist_name, 100, -0.1, 0.1);

      pmu_resolution_hist_map[{pmu, costhetamu}] = deltapmu_hist;
      costhetamu_resolution_hist_map[{pmu, costhetamu}] = deltacosthetamu_hist;

    }
  }


  double previus_Pmu = -1;;
  double previus_CosThetamu = -2;

  for (int i = 0; i < tree->GetEntries(); ++i) {
    tree->GetEntry(i);
    if(Pmu == previus_Pmu && CosThetamu == previus_CosThetamu){
      continue; // skip duplicate entries
    }
    TruePmu *= 1000.; // convert to MeV/c
    double deltapmu = (Pmu - TruePmu) / TruePmu ;
    double deltacosthetamu = CosThetamu - TrueCosThetamu;

//    cout << "Entry " << i << ": Pmu = " << Pmu << ", TruePmu = " << TruePmu
//         << ", Costhetamu = " << CosThetamu << ", TrueCosthetamu = " << TrueCosThetamu << std::endl;

    for(size_t j = 0; j < pmu_bins.size()-1; ++j){
      for(size_t k = 0; k < costhetamu_bins.size()-1; ++k){
        if(Pmu >= pmu_bins[j] && Pmu < pmu_bins[j+1] &&
          CosThetamu >= costhetamu_bins[k] && CosThetamu < costhetamu_bins[k+1]){
          pmu_resolution_hist_map[{pmu_bins[j], costhetamu_bins[k]}]->Fill(deltapmu);
          costhetamu_resolution_hist_map[{pmu_bins[j], costhetamu_bins[k]}]->Fill(deltacosthetamu);
//          cout<<"Filling histograms for Pmu bin: " << pmu_bins[j] << " to " << pmu_bins[j+1]
//              << ", Costhetamu bin: " << costhetamu_bins[k] << " to " << costhetamu_bins[k+1] << " ->  "<<deltapmu<<", "<<deltacosthetamu<<endl;
        }
      }
    }
    previus_CosThetamu = CosThetamu;
    previus_Pmu = Pmu;


  }

  TH2D* pmu_resolution_map = new TH2D("pmu_resolution_map", "PMU Resolution Map;Pmu;CosThetamu", pmu_bins.size()-1, &pmu_bins[0], costhetamu_bins.size()-1, &costhetamu_bins[0]);
  TH2D* costhetamu_resolution_map = new TH2D("costhetamu_resolution_map", "Costhetamu Resolution Map;Pmu;CosThetamu", pmu_bins.size()-1, &pmu_bins[0], costhetamu_bins.size()-1, &costhetamu_bins[0]);

  string test_folder = "test_folder/";
  gSystem->mkdir(test_folder.c_str(), true);

  // Loop over the histograms, fit Gaussian and print results
  for(auto item : pmu_resolution_hist_map) {
    TCanvas *c = new TCanvas("c", "c", 800, 600);
    TH1F* hist = item.second;
    hist->Draw();
    if (hist->GetEntries() == 0) continue; // Skip empty histograms
    TF1 *gaus_fit = new TF1("gaus_fit", "gaus");
    hist->Fit(gaus_fit, "Q");
    TH1F* hist_no_tails = (TH1F*)hist->Clone();
    for (int bin = 1; bin <= hist_no_tails->GetNbinsX(); ++bin) {
      double bin_content = hist_no_tails->GetBinContent(bin);
      if (bin_content < hist->GetMaximum() * 0.15){
        hist_no_tails->SetBinContent(bin, 0);
      }
    }
    double sigma = hist_no_tails->GetRMS();
    hist_no_tails->SetFillColor(kRed-10);
    hist_no_tails->Draw("SAME");
    c->SaveAs((test_folder + hist->GetName() + ".png").c_str());
    std::cout << item.first.first << " " << item.first.second << ", PMU Sigma: " << sigma << std::endl;
    pmu_resolution_map->Fill(
      item.first.first,
      item.first.second,
      sigma
    );

  }

  for(auto item : costhetamu_resolution_hist_map) {
    TCanvas *c = new TCanvas("c", "c", 800, 600);
    TH1F* hist = item.second;
    hist->Draw();
    if (hist->GetEntries() == 0) continue; // Skip empty histograms
    TF1 *gaus_fit = new TF1("gaus_fit", "gaus");
    hist->Fit(gaus_fit, "Q");
    TH1F* hist_no_tails = (TH1F*)hist->Clone();
    for (int bin = 1; bin <= hist_no_tails->GetNbinsX(); ++bin) {
      double bin_content = hist_no_tails->GetBinContent(bin);
      if (bin_content < hist->GetMaximum() * 0.15){
        hist_no_tails->SetBinContent(bin, 0);
      }
    }
    double sigma = hist_no_tails->GetRMS();
    hist_no_tails->SetFillColor(kRed-10);
    hist_no_tails->Draw("SAME");
    c->SaveAs((test_folder + hist->GetName() + ".png").c_str());
    std::cout << item.first.first << " " << item.first.second << ", COSTHETAMU Sigma: " << sigma << std::endl;
    costhetamu_resolution_map->Fill(
      item.first.first,
      item.first.second,
      sigma
    );
  }

  // Draw the resolution maps
  TCanvas *c1 = new TCanvas("c1", "PMU Resolution Map", 800, 600);
  pmu_resolution_map->Draw("COLZ");
  pmu_resolution_map->SetStats(0);
  c1->SetRightMargin(0.15);
  pmu_resolution_map->GetZaxis()->SetTitleOffset(1.1);
  pmu_resolution_map->GetZaxis()->SetTitle("#sigma_{P#mu}/P_{#mu}");
  pmu_resolution_map->SetTitle("P_{#mu} Resolution Map");
  c1->SetLogz();
  pmu_resolution_map->SetName("pmu_resolution_map");
  pmu_resolution_map->SaveAs("pmu_resolution_map.root");
  c1->SaveAs("pmu_resolution_map.png");
  TCanvas *c2 = new TCanvas("c2", "Costhetamu Resolution Map", 800, 600);
  costhetamu_resolution_map->Draw("COLZ");
  costhetamu_resolution_map->SetStats(0);
  c2->SetRightMargin(0.15);
  costhetamu_resolution_map->GetZaxis()->SetTitleOffset(1.3);
  costhetamu_resolution_map->GetZaxis()->SetTitle("#sigma_{Cos#theta_{mu}}");
  c2->SetLogz();
  costhetamu_resolution_map->SetTitle("Cos#theta_{#mu} Resolution Map");
  costhetamu_resolution_map->SetName("costhetamu_resolution_map");
  costhetamu_resolution_map->SaveAs("costhetamu_resolution_map.root");
  c2->SaveAs("costhetamu_resolution_map.png");


}