#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TTree.h>

#include "fastjet/PseudoJet.hh"

namespace {

constexpr int kMaxJets = 100;
constexpr double kBJetPtCut = 20.0;
constexpr double kBJetEtaCut = 2.5;

struct Counter {
  long long events = 0;
  double weight = 0.0;

  void fill(double event_weight) {
    ++events;
    weight += event_weight;
  }
};

std::string outputPrefix(const std::string& input, const char* requested) {
  if (requested && requested[0] != '\0') {
    return requested;
  }
  std::string prefix = input;
  const std::size_t dot = prefix.rfind('.');
  if (dot != std::string::npos) {
    prefix.erase(dot);
  }
  return prefix + "_merged_bjet_study";
}

double fraction(long long numerator, long long denominator) {
  return denominator > 0 ? static_cast<double>(numerator) / denominator : 0.0;
}

double weightedFraction(double numerator, double denominator) {
  return denominator != 0.0 ? numerator / denominator : 0.0;
}

void printCounter(const std::string& label, const Counter& counter,
                  const Counter& total) {
  std::cout << std::left << std::setw(42) << label
            << std::right << std::setw(10) << counter.events
            << "  " << std::setw(10) << std::fixed << std::setprecision(4)
            << 100.0 * fraction(counter.events, total.events) << "%"
            << "  weighted " << std::setw(10)
            << 100.0 * weightedFraction(counter.weight, total.weight) << "%\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2 || argc > 4) {
    std::cerr << "Usage: " << argv[0]
              << " INPUT.root [OUTPUT_PREFIX] [MAX_EVENTS]\n";
    return 1;
  }

  const std::string input = argv[1];
  const std::string output_prefix = outputPrefix(input, argc >= 3 ? argv[2] : nullptr);
  long long max_events = -1;
  if (argc == 4) {
    max_events = std::atoll(argv[3]);
    if (max_events < 1) {
      std::cerr << "MAX_EVENTS must be positive\n";
      return 1;
    }
  }

  TChain chain("Data");
  if (input.size() >= 5 && input.substr(input.size() - 5) == ".root") {
    chain.Add(input.c_str());
  } else if (input.size() >= 6 && input.substr(input.size() - 6) == ".input") {
    std::ifstream list(input.c_str());
    std::string path;
    while (list >> path) {
      chain.Add(path.c_str());
    }
  } else {
    std::cerr << "Input must be a ROOT file or a .input file list\n";
    return 1;
  }

  const long long available_events = chain.GetEntries();
  if (available_events < 1) {
    std::cerr << "No events found in " << input << "\n";
    return 1;
  }
  if (!chain.GetBranch("bHadronMultiplicity")) {
    std::cerr << "Input does not contain bHadronMultiplicity. Regenerate it "
              << "with the updated HwSim library.\n";
    return 2;
  }

  double event_weight = 0.0;
  int number_bjets = 0;
  double bjets[5][kMaxJets] = {};
  int b_hadron_multiplicity[kMaxJets] = {};
  chain.SetBranchAddress("evweight", &event_weight);
  chain.SetBranchAddress("numbJets", &number_bjets);
  chain.SetBranchAddress("thebJets", &bjets);
  chain.SetBranchAddress("bHadronMultiplicity", &b_hadron_multiplicity);

  const long long events_to_run =
      max_events > 0 ? std::min(max_events, available_events) : available_events;

  TFile output((output_prefix + ".root").c_str(), "RECREATE");
  TH1D h_number_bjets("number_bjets", "HwSim b jets per event;N_{b jets};Events", 16, -0.5, 15.5);
  TH1D h_number_accepted_bjets(
      "number_accepted_bjets",
      "Accepted b jets per event (pT > 20 GeV, |eta| < 2.5);N_{b jets};Events",
      16, -0.5, 15.5);
  TH1D h_number_truth_tags(
      "number_truth_tags",
      "Accepted b-hadron tag equivalents per event;sum N_{b hadrons};Events",
      20, -0.5, 19.5);
  TH1D h_number_merged_bjets(
      "number_merged_bjets",
      "Accepted merged b jets per event;N_{jets}(N_{b hadrons} >= 2);Events",
      9, -0.5, 8.5);
  TH1D h_b_hadron_multiplicity(
      "b_hadron_multiplicity",
      "b-hadron multiplicity in accepted b jets;N_{b hadrons};Jets",
      7, -0.5, 6.5);
  TH1D h_merged_bjet_pt(
      "merged_bjet_pt", "Merged b-jet pT;pT [GeV];Jets", 100, 0.0, 1000.0);
  TH1D h_merged_bjet_mass(
      "merged_bjet_mass", "Merged b-jet mass;m [GeV];Jets", 100, 0.0, 250.0);

  TTree event_tree("MergedBJetStudy", "Per-event merged-b-jet diagnostic");
  long long event_index = -1;
  int raw_bjets = 0;
  int raw_truth_tags = 0;
  int raw_merged_bjets = 0;
  int accepted_bjets = 0;
  int accepted_truth_tags = 0;
  int accepted_merged_bjets = 0;
  bool resolved_eight = false;
  bool tag_equivalent_eight = false;
  bool recoverable_by_multiplicity = false;
  event_tree.Branch("event_index", &event_index);
  event_tree.Branch("weight", &event_weight);
  event_tree.Branch("raw_bjets", &raw_bjets);
  event_tree.Branch("raw_truth_tags", &raw_truth_tags);
  event_tree.Branch("raw_merged_bjets", &raw_merged_bjets);
  event_tree.Branch("accepted_bjets", &accepted_bjets);
  event_tree.Branch("accepted_truth_tags", &accepted_truth_tags);
  event_tree.Branch("accepted_merged_bjets", &accepted_merged_bjets);
  event_tree.Branch("resolved_eight", &resolved_eight);
  event_tree.Branch("tag_equivalent_eight", &tag_equivalent_eight);
  event_tree.Branch("recoverable_by_multiplicity", &recoverable_by_multiplicity);

  Counter total;
  Counter any_raw_merged;
  Counter any_accepted_merged;
  Counter multiple_accepted_merged;
  Counter multiplicity_three_or_more;
  Counter resolved_eight_counter;
  Counter tag_equivalent_eight_counter;
  Counter recoverable_counter;
  long long invalid_multiplicity_entries = 0;
  long long total_accepted_merged_jets = 0;

  for (event_index = 0; event_index < events_to_run; ++event_index) {
    chain.GetEntry(event_index);
    total.fill(event_weight);

    raw_bjets = 0;
    raw_truth_tags = 0;
    raw_merged_bjets = 0;
    accepted_bjets = 0;
    accepted_truth_tags = 0;
    accepted_merged_bjets = 0;
    bool has_multiplicity_three = false;

    const int safe_number_bjets = std::max(0, std::min(number_bjets, kMaxJets));
    for (int jet_index = 0; jet_index < safe_number_bjets; ++jet_index) {
      const int stored_multiplicity = b_hadron_multiplicity[jet_index];
      if (stored_multiplicity < 1) {
        ++invalid_multiplicity_entries;
      }
      const int multiplicity = std::max(1, stored_multiplicity);
      ++raw_bjets;
      raw_truth_tags += multiplicity;
      if (multiplicity >= 2) {
        ++raw_merged_bjets;
      }

      const fastjet::PseudoJet jet(bjets[1][jet_index], bjets[2][jet_index],
                                    bjets[3][jet_index], bjets[0][jet_index]);
      // Match the truth b-jet acceptance used before smearing in the CMS analysis.
      if (jet.perp() <= kBJetPtCut || std::fabs(jet.eta()) >= kBJetEtaCut) {
        continue;
      }

      ++accepted_bjets;
      accepted_truth_tags += multiplicity;
      h_b_hadron_multiplicity.Fill(multiplicity, event_weight);
      if (multiplicity >= 2) {
        ++accepted_merged_bjets;
        ++total_accepted_merged_jets;
        h_merged_bjet_pt.Fill(jet.perp(), event_weight);
        h_merged_bjet_mass.Fill(jet.m(), event_weight);
      }
      if (multiplicity >= 3) {
        has_multiplicity_three = true;
      }
    }

    resolved_eight = accepted_bjets >= 8;
    tag_equivalent_eight = accepted_truth_tags >= 8;
    recoverable_by_multiplicity = !resolved_eight && tag_equivalent_eight;

    if (raw_merged_bjets > 0) any_raw_merged.fill(event_weight);
    if (accepted_merged_bjets > 0) any_accepted_merged.fill(event_weight);
    if (accepted_merged_bjets > 1) multiple_accepted_merged.fill(event_weight);
    if (has_multiplicity_three) multiplicity_three_or_more.fill(event_weight);
    if (resolved_eight) resolved_eight_counter.fill(event_weight);
    if (tag_equivalent_eight) tag_equivalent_eight_counter.fill(event_weight);
    if (recoverable_by_multiplicity) recoverable_counter.fill(event_weight);

    h_number_bjets.Fill(raw_bjets, event_weight);
    h_number_accepted_bjets.Fill(accepted_bjets, event_weight);
    h_number_truth_tags.Fill(accepted_truth_tags, event_weight);
    h_number_merged_bjets.Fill(accepted_merged_bjets, event_weight);
    event_tree.Fill();
  }

  output.cd();
  event_tree.Write();
  h_number_bjets.Write();
  h_number_accepted_bjets.Write();
  h_number_truth_tags.Write();
  h_number_merged_bjets.Write();
  h_b_hadron_multiplicity.Write();
  h_merged_bjet_pt.Write();
  h_merged_bjet_mass.Write();
  output.Close();

  std::cout << "\nMerged-b-jet truth study\n"
            << "Input: " << input << "\n"
            << "Acceptance: pT > " << kBJetPtCut << " GeV, |eta| < "
            << kBJetEtaCut << "\n"
            << "Events: " << total.events << ", total weight: " << total.weight
            << "\n\n";
  printCounter("At least one raw merged b jet", any_raw_merged, total);
  printCounter("At least one accepted merged b jet", any_accepted_merged, total);
  printCounter("At least two accepted merged b jets", multiple_accepted_merged, total);
  printCounter("Accepted multiplicity >= 3", multiplicity_three_or_more, total);
  printCounter("At least eight accepted b-jet objects", resolved_eight_counter, total);
  printCounter("At least eight accepted truth tag equivalents",
               tag_equivalent_eight_counter, total);
  printCounter("Recoverable by b-hadron multiplicity", recoverable_counter, total);
  std::cout << "Total accepted merged b jets: " << total_accepted_merged_jets << "\n"
            << "Invalid/unavailable multiplicity entries: "
            << invalid_multiplicity_entries << "\n";

  std::ofstream json((output_prefix + ".json").c_str());
  json << std::setprecision(12);
  json << "{\n"
       << "  \"input\": \"" << input << "\",\n"
       << "  \"events\": " << total.events << ",\n"
       << "  \"total_weight\": " << total.weight << ",\n"
       << "  \"pt_cut_gev\": " << kBJetPtCut << ",\n"
       << "  \"eta_cut\": " << kBJetEtaCut << ",\n"
       << "  \"events_with_raw_merged_bjet\": " << any_raw_merged.events << ",\n"
       << "  \"events_with_accepted_merged_bjet\": " << any_accepted_merged.events << ",\n"
       << "  \"events_with_multiple_accepted_merged_bjets\": "
       << multiple_accepted_merged.events << ",\n"
       << "  \"events_with_accepted_multiplicity_ge3\": "
       << multiplicity_three_or_more.events << ",\n"
       << "  \"events_with_eight_accepted_bjet_objects\": "
       << resolved_eight_counter.events << ",\n"
       << "  \"events_with_eight_accepted_truth_tag_equivalents\": "
       << tag_equivalent_eight_counter.events << ",\n"
       << "  \"events_recoverable_by_multiplicity\": "
       << recoverable_counter.events << ",\n"
       << "  \"fraction_with_accepted_merged_bjet\": "
       << fraction(any_accepted_merged.events, total.events) << ",\n"
       << "  \"fraction_recoverable_by_multiplicity\": "
       << fraction(recoverable_counter.events, total.events) << ",\n"
       << "  \"weighted_fraction_with_accepted_merged_bjet\": "
       << weightedFraction(any_accepted_merged.weight, total.weight) << ",\n"
       << "  \"weighted_fraction_recoverable_by_multiplicity\": "
       << weightedFraction(recoverable_counter.weight, total.weight) << ",\n"
       << "  \"total_accepted_merged_bjets\": " << total_accepted_merged_jets << ",\n"
       << "  \"invalid_multiplicity_entries\": "
       << invalid_multiplicity_entries << "\n"
       << "}\n";

  std::cout << "ROOT output: " << output_prefix << ".root\n"
            << "JSON output: " << output_prefix << ".json\n";
  return 0;
}
