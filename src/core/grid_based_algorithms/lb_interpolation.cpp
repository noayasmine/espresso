/*
 * Copyright (C) 2010-2022 The ESPResSo project
 *
 * This file is part of ESPResSo.
 *
 * ESPResSo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESPResSo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "lb_interpolation.hpp"

#include "communication.hpp"
#include "config.hpp"
#include "grid_based_algorithms/lattice.hpp"
#include "lb.hpp"

#include <utils/Vector.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace {
InterpolationOrder interpolation_order = InterpolationOrder::linear;
}

void mpi_set_interpolation_order_local(InterpolationOrder const &order) {
  interpolation_order = order;
}

REGISTER_CALLBACK(mpi_set_interpolation_order_local)

void lb_lbinterpolation_set_interpolation_order(
    InterpolationOrder const &order) {
  mpi_call_all(mpi_set_interpolation_order_local, order);
}

InterpolationOrder lb_lbinterpolation_get_interpolation_order() {
  return interpolation_order;
}

namespace {
template <typename Op>
void lattice_interpolation(Lattice const &lattice, Utils::Vector3d const &pos,
                           Op &&op) {
  Utils::Vector<std::size_t, 8> node_index{};
  Utils::Vector6d delta{};

  /* determine elementary lattice cell surrounding the particle
     and the relative position of the particle in this cell */
  lattice.map_position_to_lattice(pos, node_index, delta);
  for (int z = 0; z < 2; z++) {
    for (int y = 0; y < 2; y++) {
      for (int x = 0; x < 2; x++) {
        auto &index = node_index[(z * 2 + y) * 2 + x];
        auto const w = delta[3 * x + 0] * delta[3 * y + 1] * delta[3 * z + 2];

        op(index, w);
      }
    }
  }
}

Utils::Vector3d node_u(Lattice::index_t index) {
#ifdef LB_BOUNDARIES
  if (lbfields[index].boundary) {
    return lbfields[index].slip_velocity;
  }
#endif // LB_BOUNDARIES
  auto const modes = lb_calc_modes(index, lbfluid);
  auto const local_density = lbpar.density + modes[0];
  return Utils::Vector3d{modes[1], modes[2], modes[3]} / local_density;
}

double node_dens(Lattice::index_t index) {
#ifdef LB_BOUNDARIES
  if (lbfields[index].boundary) {
    return lbpar.density;
  }
#endif // LB_BOUNDARIES
  auto const modes = lb_calc_modes(index, lbfluid);
  return lbpar.density + modes[0];
}

#ifdef LB_BOUNDARIES
/* Velocity along one z-pair of the interpolation cell (lower node lo, upper
   node hi), at fractional height s in [0, 1] above lo.
   With halfway bounce-back the physical wall lies midway between a boundary
   node and its fluid neighbour, so the wall velocity is applied at s = 0.5
   instead of at the boundary node centre. Positions behind the wall get the
   wall velocity. Only z-walls are treated exactly (nearwall-fix patch). */
Utils::Vector3d z_pair_velocity(Lattice::index_t lo, Lattice::index_t hi,
                                double s) {
  auto const b_lo = lbfields[lo].boundary;
  auto const b_hi = lbfields[hi].boundary;
  if (b_lo and b_hi) {
    return lbfields[lo].slip_velocity;
  }
  if (b_lo) {
    auto const u_wall = lbfields[lo].slip_velocity;
    if (s <= 0.5) {
      return u_wall;
    }
    return u_wall + (node_u(hi) - u_wall) * (2. * s - 1.);
  }
  if (b_hi) {
    auto const u_wall = lbfields[hi].slip_velocity;
    if (s >= 0.5) {
      return u_wall;
    }
    return u_wall + (node_u(lo) - u_wall) * (1. - 2. * s);
  }
  return (1. - s) * node_u(lo) + s * node_u(hi);
}
#endif // LB_BOUNDARIES

} // namespace

const Utils::Vector3d
lb_lbinterpolation_get_interpolated_velocity(const Utils::Vector3d &pos) {
  Utils::Vector3d interpolated_u{};

#ifdef LB_BOUNDARIES
  /* Bilinear in x and y, wall-aware linear in z (see z_pair_velocity). */
  Utils::Vector<std::size_t, 8> node_index{};
  Utils::Vector6d delta{};
  lblattice.map_position_to_lattice(pos, node_index, delta);
  for (int y = 0; y < 2; y++) {
    for (int x = 0; x < 2; x++) {
      auto const w_xy = delta[3 * x + 0] * delta[3 * y + 1];
      auto const lo = static_cast<Lattice::index_t>(node_index[y * 2 + x]);
      auto const hi = static_cast<Lattice::index_t>(node_index[4 + y * 2 + x]);
      interpolated_u += w_xy * z_pair_velocity(lo, hi, delta[5]);
    }
  }
#else
  /* Calculate fluid velocity at particle's position.
     This is done by linear interpolation (eq. (11) @cite ahlrichs99a) */
  lattice_interpolation(lblattice, pos,
                        [&interpolated_u](Lattice::index_t index, double w) {
                          interpolated_u += w * node_u(index);
                        });
#endif // LB_BOUNDARIES

  return interpolated_u;
}

double lb_lbinterpolation_get_interpolated_density(const Utils::Vector3d &pos) {
  double interpolated_dens = 0.;

  /* Calculate fluid density at the position.
     This is done by linear interpolation (eq. (11) @cite ahlrichs99a) */
  lattice_interpolation(lblattice, pos,
                        [&interpolated_dens](Lattice::index_t index, double w) {
                          interpolated_dens += w * node_dens(index);
                        });

  return interpolated_dens;
}

void lb_lbinterpolation_add_force_density(
    const Utils::Vector3d &pos, const Utils::Vector3d &force_density) {
  switch (interpolation_order) {
  case (InterpolationOrder::quadratic):
    throw std::runtime_error("The non-linear interpolation scheme is not "
                             "implemented for the CPU LB.");
  case (InterpolationOrder::linear):
    lattice_interpolation(lblattice, pos,
                          [&force_density](Lattice::index_t index, double w) {
                            auto &field = lbfields[index];
                            field.force_density += w * force_density;
                          });
    break;
  }
}
