#ifndef QUADRUPLE_HIGGS_EXTENDED91OBSERVABLES_H
#define QUADRUPLE_HIGGS_EXTENDED91OBSERVABLES_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace extended91 {

constexpr int kJetCount = 8;
constexpr int kCandidateCount = 4;
constexpr int kPairingCount = 105;
constexpr int kFeatureCount = 91;
constexpr double kPi = 3.141592653589793238462643383279502884;

using ConstituentPair = std::array<int, 2>;
using CanonicalPairing = std::array<ConstituentPair, kCandidateCount>;

inline void makePairingsRecursive(const std::vector<int>& remaining,
                                  CanonicalPairing& current,
                                  int depth,
                                  std::vector<CanonicalPairing>& output) {
  if (remaining.empty()) {
    output.push_back(current);
    return;
  }

  const int first = remaining.front();
  for (std::size_t partner_index = 1; partner_index < remaining.size(); ++partner_index) {
    const int second = remaining[partner_index];
    current[depth] = {{std::min(first, second), std::max(first, second)}};

    std::vector<int> next;
    next.reserve(remaining.size() - 2);
    for (std::size_t index = 1; index < remaining.size(); ++index) {
      if (index != partner_index) {
        next.push_back(remaining[index]);
      }
    }
    makePairingsRecursive(next, current, depth + 1, output);
  }
}

inline std::vector<CanonicalPairing> makeCanonicalPairings() {
  std::vector<int> remaining(kJetCount);
  std::iota(remaining.begin(), remaining.end(), 0);

  CanonicalPairing current = {};
  std::vector<CanonicalPairing> output;
  output.reserve(kPairingCount);
  makePairingsRecursive(remaining, current, 0, output);
  return output;
}

template <typename FourVector>
struct Candidate {
  FourVector momentum;
  ConstituentPair constituents = {{-1, -1}};
};

template <typename FourVector>
struct Reconstruction {
  double chi8 = std::numeric_limits<double>::infinity();
  double chi8_second = std::numeric_limits<double>::infinity();
  int n_pairings_chi8_lt60 = 0;
  CanonicalPairing pairing = {};
  CanonicalPairing pairing_second = {};
  std::array<double, kCandidateCount> delta_m = {};
  std::array<FourVector, kCandidateCount> higgses = {};
  std::array<FourVector, kJetCount> canonical_jets = {};
};

template <typename FourVector>
struct FourVectorOrder {
  bool operator()(const FourVector& left, const FourVector& right) const {
    if (left.perp() != right.perp()) return left.perp() > right.perp();
    if (left.px() != right.px()) return left.px() > right.px();
    if (left.py() != right.py()) return left.py() > right.py();
    if (left.pz() != right.pz()) return left.pz() > right.pz();
    if (left.e() != right.e()) return left.e() > right.e();
    return false;
  }
};

template <typename FourVector>
struct CandidatePtOrder {
  bool operator()(const Candidate<FourVector>& left, const Candidate<FourVector>& right) const {
    const double left_pt = left.momentum.perp();
    const double right_pt = right.momentum.perp();
    if (left_pt > right_pt) {
      return true;
    }
    if (left_pt < right_pt) {
      return false;
    }
    return left.constituents < right.constituents;
  }
};

template <typename FourVector>
struct RankedReconstruction {
  double chi8 = std::numeric_limits<double>::infinity();
  CanonicalPairing pairing = {};
  std::array<double, kCandidateCount> delta_m = {};
  std::array<FourVector, kCandidateCount> higgses = {};
};

template <typename FourVector>
struct ReconstructionOrder {
  bool operator()(const RankedReconstruction<FourVector>& left,
                  const RankedReconstruction<FourVector>& right) const {
    if (left.chi8 < right.chi8) {
      return true;
    }
    if (left.chi8 > right.chi8) {
      return false;
    }
    return left.pairing < right.pairing;
  }
};

// Reconstruct the four candidates for every canonical perfect matching.  Within
// each matching the candidates are first ordered by descending pT (then by the
// canonical constituent labels), and only then associated with the four mass
// targets.  The same ordered candidates and constituents are retained in the
// returned object, preventing feature-to-candidate misalignment.
template <typename FourVector>
Reconstruction<FourVector> reconstruct(
    const std::vector<FourVector>& jets,
    const std::vector<CanonicalPairing>& pairings,
    const std::array<double, kCandidateCount>& mass_targets,
    double good_pairing_threshold = 60.0) {
  Reconstruction<FourVector> result;
  if (jets.size() != kJetCount) {
    return result;
  }

  // Canonicalize the input labels with a physical four-momentum key before
  // any constituent-index tie-break is used.  Exact duplicate four-vectors
  // are physically interchangeable, so their residual order cannot alter an
  // observable.  This makes target assignment and best-pairing selection
  // invariant under permutations of the input jet container, including exact
  // candidate-pT ties.
  std::vector<FourVector> canonical_jets = jets;
  std::sort(canonical_jets.begin(), canonical_jets.end(), FourVectorOrder<FourVector>());
  std::copy(canonical_jets.begin(), canonical_jets.end(), result.canonical_jets.begin());

  std::vector<RankedReconstruction<FourVector> > ranked;
  ranked.reserve(pairings.size());
  for (const CanonicalPairing& input_pairing : pairings) {
    std::array<Candidate<FourVector>, kCandidateCount> candidates;
    for (int candidate_index = 0; candidate_index < kCandidateCount; ++candidate_index) {
      const ConstituentPair& pair = input_pairing[candidate_index];
      candidates[candidate_index].constituents = pair;
      candidates[candidate_index].momentum =
          canonical_jets[pair[0]] + canonical_jets[pair[1]];
    }
    std::sort(candidates.begin(), candidates.end(), CandidatePtOrder<FourVector>());

    RankedReconstruction<FourVector> reconstruction;
    double chi8_squared = 0.0;
    for (int candidate_index = 0; candidate_index < kCandidateCount; ++candidate_index) {
      reconstruction.pairing[candidate_index] = candidates[candidate_index].constituents;
      reconstruction.higgses[candidate_index] = candidates[candidate_index].momentum;
      reconstruction.delta_m[candidate_index] =
          std::fabs(candidates[candidate_index].momentum.m() - mass_targets[candidate_index]);
      chi8_squared += reconstruction.delta_m[candidate_index] *
                      reconstruction.delta_m[candidate_index];
    }
    reconstruction.chi8 = std::sqrt(chi8_squared);
    if (!std::isfinite(reconstruction.chi8)) {
      reconstruction.chi8 = std::numeric_limits<double>::infinity();
    }
    ranked.push_back(reconstruction);
  }

  std::sort(ranked.begin(), ranked.end(), ReconstructionOrder<FourVector>());
  for (const RankedReconstruction<FourVector>& reconstruction : ranked) {
    if (reconstruction.chi8 < good_pairing_threshold) {
      ++result.n_pairings_chi8_lt60;
    }
  }
  if (!ranked.empty()) {
    result.chi8 = ranked[0].chi8;
    result.pairing = ranked[0].pairing;
    result.delta_m = ranked[0].delta_m;
    result.higgses = ranked[0].higgses;
  }
  if (ranked.size() > 1) {
    result.chi8_second = ranked[1].chi8;
    result.pairing_second = ranked[1].pairing;
  }
  return result;
}

inline double absoluteDeltaPhi(double phi1, double phi2) {
  double difference = std::fmod(phi1 - phi2, 2.0 * kPi);
  if (difference > kPi) {
    difference -= 2.0 * kPi;
  } else if (difference < -kPi) {
    difference += 2.0 * kPi;
  }
  return std::fabs(difference);
}

struct CartesianFourVector {
  CartesianFourVector(double px_value = 0.0,
                      double py_value = 0.0,
                      double pz_value = 0.0,
                      double energy_value = 0.0)
      : px(px_value), py(py_value), pz(pz_value), energy(energy_value) {}

  double px;
  double py;
  double pz;
  double energy;
};

inline bool boostToRestFrame(const CartesianFourVector& momentum,
                             const CartesianFourVector& parent,
                             CartesianFourVector& boosted) {
  if (!(parent.energy > 0.0) || !std::isfinite(parent.energy)) {
    return false;
  }
  const double beta_x = -parent.px / parent.energy;
  const double beta_y = -parent.py / parent.energy;
  const double beta_z = -parent.pz / parent.energy;
  const double beta_squared =
      beta_x * beta_x + beta_y * beta_y + beta_z * beta_z;
  if (!(beta_squared >= 0.0) || beta_squared >= 1.0 || !std::isfinite(beta_squared)) {
    return false;
  }

  const double gamma = 1.0 / std::sqrt(1.0 - beta_squared);
  const double beta_dot_p =
      beta_x * momentum.px + beta_y * momentum.py + beta_z * momentum.pz;
  const double gamma2 = beta_squared > 0.0 ? (gamma - 1.0) / beta_squared : 0.0;
  const double spatial_coefficient = gamma2 * beta_dot_p + gamma * momentum.energy;
  boosted.px = momentum.px + spatial_coefficient * beta_x;
  boosted.py = momentum.py + spatial_coefficient * beta_y;
  boosted.pz = momentum.pz + spatial_coefficient * beta_z;
  boosted.energy = gamma * (momentum.energy + beta_dot_p);
  return std::isfinite(boosted.px) && std::isfinite(boosted.py) &&
         std::isfinite(boosted.pz) && std::isfinite(boosted.energy);
}

// |cos(theta*)| between the decay constituent in the candidate rest frame and
// the candidate flight direction in the four-candidate rest frame.
inline double absoluteHelicityCosine(const CartesianFourVector& constituent,
                                     const CartesianFourVector& candidate,
                                     const CartesianFourVector& four_candidate,
                                     bool& defined) {
  defined = false;
  CartesianFourVector constituent_in_candidate_rest;
  CartesianFourVector candidate_in_four_candidate_rest;
  if (!boostToRestFrame(constituent, candidate, constituent_in_candidate_rest) ||
      !boostToRestFrame(candidate, four_candidate, candidate_in_four_candidate_rest)) {
    return 0.0;
  }

  const double decay_norm = std::sqrt(
      constituent_in_candidate_rest.px * constituent_in_candidate_rest.px +
      constituent_in_candidate_rest.py * constituent_in_candidate_rest.py +
      constituent_in_candidate_rest.pz * constituent_in_candidate_rest.pz);
  const double flight_norm = std::sqrt(
      candidate_in_four_candidate_rest.px * candidate_in_four_candidate_rest.px +
      candidate_in_four_candidate_rest.py * candidate_in_four_candidate_rest.py +
      candidate_in_four_candidate_rest.pz * candidate_in_four_candidate_rest.pz);
  if (!(decay_norm > 0.0) || !(flight_norm > 0.0) ||
      !std::isfinite(decay_norm) || !std::isfinite(flight_norm)) {
    return 0.0;
  }

  const double dot = constituent_in_candidate_rest.px * candidate_in_four_candidate_rest.px +
                     constituent_in_candidate_rest.py * candidate_in_four_candidate_rest.py +
                     constituent_in_candidate_rest.pz * candidate_in_four_candidate_rest.pz;
  const double cosine = std::fabs(dot / (decay_norm * flight_norm));
  if (!std::isfinite(cosine)) {
    return 0.0;
  }
  defined = true;
  return std::max(0.0, std::min(1.0, cosine));
}

inline double boundedRatio(double numerator, double denominator) {
  if (!(denominator > 0.0) || !std::isfinite(numerator) || !std::isfinite(denominator)) {
    return 0.0;
  }
  return numerator / denominator;
}

inline double momentumBalanceFraction(double first_pt, double second_pt) {
  return boundedRatio(std::min(first_pt, second_pt), first_pt + second_pt);
}

inline double centrality(const std::vector<std::pair<double, double> >& pt_and_energy) {
  double scalar_pt = 0.0;
  double scalar_energy = 0.0;
  for (const std::pair<double, double>& values : pt_and_energy) {
    scalar_pt += values.first;
    scalar_energy += values.second;
  }
  return std::max(0.0, std::min(1.0, boundedRatio(scalar_pt, scalar_energy)));
}

inline double minimumMassDistance(const std::vector<double>& masses, double reference_mass) {
  double result = std::numeric_limits<double>::infinity();
  for (double mass : masses) {
    result = std::min(result, std::fabs(mass - reference_mass));
  }
  return masses.empty() ? 0.0 : result;
}

inline double transverseSphericity(const std::vector<std::pair<double, double> >& transverse_momenta) {
  double xx = 0.0;
  double xy = 0.0;
  double yy = 0.0;
  for (const std::pair<double, double>& momentum : transverse_momenta) {
    xx += momentum.first * momentum.first;
    xy += momentum.first * momentum.second;
    yy += momentum.second * momentum.second;
  }
  const double trace = xx + yy;
  if (!(trace > 0.0) || !std::isfinite(trace)) {
    return 0.0;
  }
  const double discriminant =
      std::sqrt(std::max(0.0, (xx - yy) * (xx - yy) + 4.0 * xy * xy));
  const double value = 1.0 - discriminant / trace;
  return std::max(0.0, std::min(1.0, value));
}

template <std::size_t N>
std::array<double, 3> massSummary(const std::array<double, N>& masses,
                                  double reference_mass = 125.0) {
  static_assert(N > 0, "massSummary requires at least one mass");
  const double mean = std::accumulate(masses.begin(), masses.end(), 0.0) /
                      static_cast<double>(N);
  double variance = 0.0;
  double maximum_absolute_deviation = 0.0;
  for (double mass : masses) {
    variance += (mass - mean) * (mass - mean);
    maximum_absolute_deviation =
        std::max(maximum_absolute_deviation, std::fabs(mass - reference_mass));
  }
  variance /= static_cast<double>(N);
  return {{mean, std::sqrt(std::max(0.0, variance)), maximum_absolute_deviation}};
}

template <typename FourVector, std::size_t N>
double subsystemMass(const std::array<FourVector, kCandidateCount>& candidates,
                     const std::array<int, N>& indices) {
  static_assert(N > 0, "subsystemMass requires at least one candidate");
  FourVector total = candidates[indices[0]];
  for (std::size_t index = 1; index < N; ++index) {
    total = total + candidates[indices[index]];
  }
  return total.m();
}

}  // namespace extended91

#endif  // QUADRUPLE_HIGGS_EXTENDED91OBSERVABLES_H
