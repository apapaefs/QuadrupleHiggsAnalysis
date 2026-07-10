#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Extended91Observables.h"

namespace {

struct ToyVector {
  ToyVector(double px_value = 0.0,
            double py_value = 0.0,
            double pz_value = 0.0,
            double energy_value = 0.0)
      : px_value(px_value),
        py_value(py_value),
        pz_value(pz_value),
        energy_value(energy_value) {}

  double perp() const { return std::sqrt(px_value * px_value + py_value * py_value); }
  double m() const {
    return std::sqrt(std::max(0.0, energy_value * energy_value - px_value * px_value -
                                      py_value * py_value - pz_value * pz_value));
  }
  double px() const { return px_value; }
  double py() const { return py_value; }
  double pz() const { return pz_value; }
  double e() const { return energy_value; }

  ToyVector operator+(const ToyVector& other) const {
    return ToyVector(px_value + other.px_value, py_value + other.py_value,
                     pz_value + other.pz_value, energy_value + other.energy_value);
  }

  double px_value;
  double py_value;
  double pz_value;
  double energy_value;
};

bool close(double left, double right, double tolerance = 1.0e-10) {
  return std::fabs(left - right) <= tolerance * std::max(1.0, std::max(std::fabs(left), std::fabs(right)));
}

std::string pairingKey(const extended91::CanonicalPairing& pairing) {
  std::ostringstream output;
  for (const extended91::ConstituentPair& pair : pairing) {
    output << pair[0] << ":" << pair[1] << ";";
  }
  return output.str();
}

ToyVector massless(double pt, double phi, double eta) {
  const double px = pt * std::cos(phi);
  const double py = pt * std::sin(phi);
  const double pz = pt * std::sinh(eta);
  return ToyVector(px, py, pz, std::sqrt(px * px + py * py + pz * pz));
}

extended91::CartesianFourVector boostByVelocity(
    const extended91::CartesianFourVector& momentum,
    double beta_x,
    double beta_y,
    double beta_z) {
  const double beta_squared =
      beta_x * beta_x + beta_y * beta_y + beta_z * beta_z;
  assert(beta_squared < 1.0);
  const double gamma = 1.0 / std::sqrt(1.0 - beta_squared);
  const double beta_dot_p =
      beta_x * momentum.px + beta_y * momentum.py + beta_z * momentum.pz;
  const double gamma2 = beta_squared > 0.0 ? (gamma - 1.0) / beta_squared : 0.0;
  const double coefficient = gamma2 * beta_dot_p + gamma * momentum.energy;
  return extended91::CartesianFourVector(
      momentum.px + coefficient * beta_x,
      momentum.py + coefficient * beta_y,
      momentum.pz + coefficient * beta_z,
      gamma * (momentum.energy + beta_dot_p));
}

}  // namespace

int main() {
  const std::vector<extended91::CanonicalPairing> pairings =
      extended91::makeCanonicalPairings();
  assert(pairings.size() == 105);
  std::set<std::string> unique_pairings;
  for (const extended91::CanonicalPairing& pairing : pairings) {
    std::array<bool, 8> used = {};
    for (const extended91::ConstituentPair& pair : pairing) {
      assert(pair[0] < pair[1]);
      assert(!used[pair[0]] && !used[pair[1]]);
      used[pair[0]] = true;
      used[pair[1]] = true;
    }
    assert(std::find(used.begin(), used.end(), false) == used.end());
    unique_pairings.insert(pairingKey(pairing));
  }
  assert(unique_pairings.size() == 105);

  const std::array<double, 4> targets = {{120.0, 115.0, 110.0, 105.0}};
  const std::vector<ToyVector> tied_jets(8, ToyVector(50.0, 0.0, 0.0, 50.0));
  const extended91::Reconstruction<ToyVector> tied =
      extended91::reconstruct(tied_jets, pairings, targets);
  const extended91::CanonicalPairing expected_first = {{
      {{0, 1}}, {{2, 3}}, {{4, 5}}, {{6, 7}},
  }};
  const extended91::CanonicalPairing expected_second = {{
      {{0, 1}}, {{2, 3}}, {{4, 6}}, {{5, 7}},
  }};
  assert(tied.pairing == expected_first);
  assert(tied.pairing_second == expected_second);
  assert(close(tied.chi8, tied.chi8_second));

  const std::array<double, 8> pts = {{140.0, 125.0, 110.0, 95.0, 80.0, 70.0, 60.0, 50.0}};
  const std::array<double, 8> phis = {{0.0, 0.8, 1.7, 2.5, -2.8, -1.9, -1.0, -0.2}};
  const std::array<double, 8> etas = {{0.2, -0.3, 0.7, -0.8, 1.1, -1.2, 0.5, -0.6}};
  std::vector<ToyVector> jets;
  for (int index = 0; index < 8; ++index) {
    jets.push_back(massless(pts[index], phis[index], etas[index]));
  }
  const extended91::Reconstruction<ToyVector> nominal =
      extended91::reconstruct(jets, pairings, targets);
  const std::array<int, 8> permutation = {{5, 1, 7, 3, 0, 6, 2, 4}};
  std::vector<ToyVector> permuted;
  for (int index : permutation) {
    permuted.push_back(jets[index]);
  }
  const extended91::Reconstruction<ToyVector> shuffled =
      extended91::reconstruct(permuted, pairings, targets);
  assert(close(nominal.chi8, shuffled.chi8));
  assert(close(nominal.chi8_second, shuffled.chi8_second));
  assert(nominal.n_pairings_chi8_lt60 == shuffled.n_pairings_chi8_lt60);
  assert(nominal.chi8_second >= nominal.chi8);
  assert(nominal.n_pairings_chi8_lt60 >= 0 && nominal.n_pairings_chi8_lt60 <= 105);
  for (int candidate = 0; candidate < 4; ++candidate) {
    if (candidate > 0) {
      assert(nominal.higgses[candidate - 1].perp() >= nominal.higgses[candidate].perp());
    }
    const extended91::ConstituentPair& pair = nominal.pairing[candidate];
    assert(pair[0] < pair[1]);
    const ToyVector aligned =
        nominal.canonical_jets[pair[0]] + nominal.canonical_jets[pair[1]];
    assert(close(nominal.higgses[candidate].perp(), aligned.perp()));
    assert(close(nominal.higgses[candidate].m(), aligned.m()));
    assert(close(nominal.higgses[candidate].perp(), shuffled.higgses[candidate].perp()));
    assert(close(nominal.higgses[candidate].m(), shuffled.higgses[candidate].m()));
    assert(close(nominal.delta_m[candidate], shuffled.delta_m[candidate]));
  }

  // Exact candidate-pT ties must not expose input-container labels to the
  // staggered target assignment.  These four physical back-to-back pairs have
  // masses exactly equal to the four targets and candidate pT=0.
  std::vector<ToyVector> tied_candidate_jets;
  for (double pt : std::array<double, 4>{{60.0, 57.5, 55.0, 52.5}}) {
    tied_candidate_jets.push_back(massless(pt, 0.0, 0.0));
    tied_candidate_jets.push_back(massless(pt, extended91::kPi, 0.0));
  }
  const extended91::Reconstruction<ToyVector> tied_candidate_nominal =
      extended91::reconstruct(tied_candidate_jets, pairings, targets);
  const std::array<int, 8> tied_candidate_permutation = {{4, 5, 0, 1, 6, 7, 2, 3}};
  std::vector<ToyVector> tied_candidate_shuffled;
  for (int index : tied_candidate_permutation) {
    tied_candidate_shuffled.push_back(tied_candidate_jets[index]);
  }
  const extended91::Reconstruction<ToyVector> tied_candidate_permuted =
      extended91::reconstruct(tied_candidate_shuffled, pairings, targets);
  assert(close(tied_candidate_nominal.chi8, 0.0));
  assert(close(tied_candidate_nominal.chi8, tied_candidate_permuted.chi8));
  assert(close(tied_candidate_nominal.chi8_second, tied_candidate_permuted.chi8_second));

  assert(close(extended91::absoluteDeltaPhi(0.2, 2.0 * extended91::kPi - 0.1), 0.3));
  const std::vector<std::pair<double, double> > pencil = {{1.0, 0.0}, {-1.0, 0.0}};
  const std::vector<std::pair<double, double> > isotropic = {{1.0, 0.0}, {0.0, 1.0}};
  assert(close(extended91::transverseSphericity(pencil), 0.0));
  assert(close(extended91::transverseSphericity(isotropic), 1.0));

  const std::array<double, 4> masses = {{100.0, 110.0, 120.0, 130.0}};
  const std::array<double, 3> summary = extended91::massSummary(masses);
  assert(close(summary[0], 115.0));
  assert(close(summary[1], std::sqrt(125.0)));
  assert(close(summary[2], 25.0));
  assert(close(extended91::momentumBalanceFraction(30.0, 70.0), 0.3));
  const std::vector<std::pair<double, double> > pt_and_energy = {{3.0, 5.0}, {4.0, 5.0}};
  assert(close(extended91::centrality(pt_and_energy), 0.7));
  const std::vector<double> pair_masses = {{80.0, 100.0, 140.0}};
  assert(close(extended91::minimumMassDistance(pair_masses, 91.1876), 8.8124));
  const std::array<ToyVector, 4> rest_candidates = {{
      ToyVector(0.0, 0.0, 0.0, 10.0), ToyVector(0.0, 0.0, 0.0, 20.0),
      ToyVector(0.0, 0.0, 0.0, 30.0), ToyVector(0.0, 0.0, 0.0, 40.0),
  }};
  assert(close(extended91::subsystemMass(rest_candidates, std::array<int, 2>{{0, 1}}), 30.0));
  assert(close(extended91::subsystemMass(rest_candidates, std::array<int, 3>{{0, 2, 3}}), 80.0));

  const double higgs_mass = 125.0;
  const double higgs_pz = 100.0;
  const double higgs_energy = std::sqrt(higgs_mass * higgs_mass + higgs_pz * higgs_pz);
  const double beta = higgs_pz / higgs_energy;
  const double gamma = 1.0 / std::sqrt(1.0 - beta * beta);
  const double decay_momentum = higgs_mass / 2.0;
  const extended91::CartesianFourVector higgs(0.0, 0.0, higgs_pz, higgs_energy);
  const extended91::CartesianFourVector four_higgs(0.0, 0.0, 0.0, 600.0);
  const extended91::CartesianFourVector parallel_constituent(
      0.0, 0.0, gamma * decay_momentum * (1.0 + beta),
      gamma * decay_momentum * (1.0 + beta));
  const extended91::CartesianFourVector perpendicular_constituent(
      decay_momentum, 0.0, gamma * beta * decay_momentum, gamma * decay_momentum);
  bool defined = false;
  assert(close(extended91::absoluteHelicityCosine(
                   parallel_constituent, higgs, four_higgs, defined),
               1.0));
  assert(defined);
  assert(close(extended91::absoluteHelicityCosine(
                   perpendicular_constituent, higgs, four_higgs, defined),
               0.0));
  assert(defined);

  // The helicity angle is a decay-frame observable and must be invariant
  // under a common, non-collinear boost of the complete event.
  const double common_beta_x = 0.31;
  const double common_beta_y = -0.17;
  const double common_beta_z = 0.08;
  const extended91::CartesianFourVector boosted_higgs = boostByVelocity(
      higgs, common_beta_x, common_beta_y, common_beta_z);
  const extended91::CartesianFourVector boosted_four_higgs = boostByVelocity(
      four_higgs, common_beta_x, common_beta_y, common_beta_z);
  const extended91::CartesianFourVector boosted_parallel = boostByVelocity(
      parallel_constituent, common_beta_x, common_beta_y, common_beta_z);
  const extended91::CartesianFourVector boosted_perpendicular = boostByVelocity(
      perpendicular_constituent, common_beta_x, common_beta_y, common_beta_z);
  assert(close(extended91::absoluteHelicityCosine(
                   boosted_parallel, boosted_higgs, boosted_four_higgs, defined),
               1.0, 1.0e-9));
  assert(defined);
  assert(close(extended91::absoluteHelicityCosine(
                   boosted_perpendicular, boosted_higgs, boosted_four_higgs, defined),
               0.0, 1.0e-9));
  assert(defined);

  return 0;
}
