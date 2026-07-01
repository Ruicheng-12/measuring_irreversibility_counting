#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <omp.h>
#include <iomanip>

// Constants and parameters
const double L = 5;
int N;
const double k_BT = 1.0;
const double epsilon = 1.0;
const double k_spring = 5.0;
const double r0 = 0.001;
const double CUTOFF = 2;
const double dt = 0.0002;
const double T = 1.0; // period of the electric force
const int equilibration_steps = 3000000;
const int production_steps = 7200000;
const double friction_coeff = 1.0;
const int grid_size = 5;
const int num_regions = grid_size * grid_size;
const std::vector<double> E_field_steps = {0.0, 0.1, 0.2, 0.4, 0.5, 0.8, 1.0};
const std::vector<double> sigma_values = {0.01, 0.02, 0.03};
const std::vector<double> charge_values = {0.9, 1.0, 1.1};

// Phase-locked sampling.  For E(t)=E+E sin(2 pi t/T), t0=T/4 is the
// maximum of the protocol.  Windows [t0+kT, t0+kT+ell T] are invariant
// under time reversal of the protocol.
const double sampling_phase = 0.25; // t0/T

inline bool is_valid_number(double x) {
    return !std::isnan(x) && !std::isinf(x) && std::abs(x) < 1e6;
}

inline double electric_field_value(double E, double t, double period) {
    return E + E * std::sin(2.0 * M_PI * t / period);
}

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> uniform_dist(0.0, L);
std::uniform_int_distribution<> sigma_dist(0, sigma_values.size() - 1);
std::uniform_int_distribution<> charge_dist(0, charge_values.size() - 1);

double minimum_image_distance(double delta, double box_length) {
    delta -= box_length * std::round(delta / box_length);
    return delta;
}

void initialize_particles(std::vector<std::vector<double>> &positions,
                          std::vector<double> &sigmas,
                          std::vector<double> &diffusion_coefficients,
                          std::vector<double> &charges) {
    positions.resize(N, std::vector<double>(2));
    sigmas.resize(N);
    diffusion_coefficients.resize(N);
    charges.resize(N);

    double max_sigma = *std::max_element(sigma_values.begin(), sigma_values.end());
    for (int i = 0; i < N; ++i) {
        positions[i][0] = uniform_dist(gen);
        positions[i][1] = uniform_dist(gen);
        sigmas[i] = sigma_values[sigma_dist(gen)];
        diffusion_coefficients[i] = max_sigma / sigmas[i];
        charges[i] = charge_values[charge_dist(gen)];
    }
}

void compute_wca_force(const std::vector<std::vector<double>> &positions,
                       const std::vector<double> &sigmas,
                       std::vector<std::vector<double>> &forces) {
    if (N > 2) {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                double dx = minimum_image_distance(positions[i][0] - positions[j][0], L);
                double dy = minimum_image_distance(positions[i][1] - positions[j][1], L);
                double r2 = dx * dx + dy * dy;
                if (r2 < 1e-10) r2 = 1e-10;
                double sigma = 0.5 * (sigmas[i] + sigmas[j]);
                if (r2 < std::pow(2.0, 1.0 / 3.0) * sigma * sigma) {
                    double r2_inv = 1.0 / r2;
                    double r6_inv = std::pow(r2_inv, 3);
                    double force_scalar = std::min(24.0 * epsilon * r6_inv * (2.0 * r6_inv - 1.0) * r2_inv, 1e3);
                    if (is_valid_number(force_scalar)) {
                        #pragma omp atomic
                        forces[i][0] += force_scalar * dx;
                        #pragma omp atomic
                        forces[i][1] += force_scalar * dy;
                        #pragma omp atomic
                        forces[j][0] -= force_scalar * dx;
                        #pragma omp atomic
                        forces[j][1] -= force_scalar * dy;
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                double dx = minimum_image_distance(positions[i][0] - positions[j][0], L);
                double dy = minimum_image_distance(positions[i][1] - positions[j][1], L);
                double r2 = dx * dx + dy * dy;
                if (r2 < 1e-10) r2 = 1e-10;
                double sigma = 0.5 * (sigmas[i] + sigmas[j]);
                if (r2 < std::pow(2.0, 1.0 / 3.0) * sigma * sigma) {
                    double r2_inv = 1.0 / r2;
                    double r6_inv = std::pow(r2_inv, 3);
                    double force_scalar = std::min(24.0 * epsilon * r6_inv * (2.0 * r6_inv - 1.0) * r2_inv, 1e3);
                    if (is_valid_number(force_scalar)) {
                        forces[i][0] += force_scalar * dx;
                        forces[i][1] += force_scalar * dy;
                        forces[j][0] -= force_scalar * dx;
                        forces[j][1] -= force_scalar * dy;
                    }
                }
            }
        }
    }
}

void compute_spring_force(const std::vector<std::vector<double>> &positions,
                          std::vector<std::vector<double>> &forces) {
    if (N > 2) {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                double dx = minimum_image_distance(positions[i][0] - positions[j][0], L);
                double dy = minimum_image_distance(positions[i][1] - positions[j][1], L);
                double r = std::sqrt(dx * dx + dy * dy);
                if (r > 1e-10 && r < CUTOFF) {
                    double force_scalar = std::min(std::max(-k_spring * (r - r0) / r, -1e3), 1e3);
                    if (is_valid_number(force_scalar)) {
                        #pragma omp atomic
                        forces[i][0] += force_scalar * dx;
                        #pragma omp atomic
                        forces[i][1] += force_scalar * dy;
                        #pragma omp atomic
                        forces[j][0] -= force_scalar * dx;
                        #pragma omp atomic
                        forces[j][1] -= force_scalar * dy;
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                double dx = minimum_image_distance(positions[i][0] - positions[j][0], L);
                double dy = minimum_image_distance(positions[i][1] - positions[j][1], L);
                double r = std::sqrt(dx * dx + dy * dy);
                // Minimal correction: apply the same finite cutoff also for N <= 2.
                if (r > 1e-10 && r < CUTOFF) {
                    double force_scalar = std::min(std::max(-k_spring * (r - r0) / r, -1e3), 1e3);
                    if (is_valid_number(force_scalar)) {
                        forces[i][0] += force_scalar * dx;
                        forces[i][1] += force_scalar * dy;
                        forces[j][0] -= force_scalar * dx;
                        forces[j][1] -= force_scalar * dy;
                    }
                }
            }
        }
    }
}

void compute_electric_force(const std::vector<double> &charges,
                            std::vector<std::vector<double>> &forces,
                            double E, double t, double period) {
    double E_t = electric_field_value(E, t, period);
    if (N > 2) {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double force = std::min(std::max(charges[i] * E_t, -1e3), 1e3);
            if (is_valid_number(force)) {
                #pragma omp atomic
                forces[i][0] += force;
            }
        }
    } else {
        for (int i = 0; i < N; ++i) {
            double force = std::min(std::max(charges[i] * E_t, -1e3), 1e3);
            if (is_valid_number(force)) forces[i][0] += force;
        }
    }
}

void langevin_step(std::vector<std::vector<double>> &positions,
                   const std::vector<std::vector<double>> &forces,
                   const std::vector<double> &diffusion_coefficients) {
    #pragma omp parallel
    {
        std::mt19937 local_gen(rd() + omp_get_thread_num() * 1000);
        std::normal_distribution<double> local_normal(0.0, 1.0);
        #pragma omp for
        for (int i = 0; i < N; ++i) {
            double noise_x = std::min(std::max(
                std::sqrt(2.0 * diffusion_coefficients[i] * dt / friction_coeff) * local_normal(local_gen),
                -0.1), 0.1);
            double noise_y = std::min(std::max(
                std::sqrt(2.0 * diffusion_coefficients[i] * dt / friction_coeff) * local_normal(local_gen),
                -0.1), 0.1);
            double dx = std::min(std::max((diffusion_coefficients[i] * forces[i][0]) * dt, -1.0), 1.0);
            double dy = std::min(std::max((diffusion_coefficients[i] * forces[i][1]) * dt, -1.0), 1.0);
            if (is_valid_number(dx + noise_x) && is_valid_number(dy + noise_y)) {
                positions[i][0] += dx + noise_x;
                positions[i][1] += dy + noise_y;
                positions[i][0] = std::fmod(positions[i][0] + L, L);
                positions[i][1] = std::fmod(positions[i][1] + L, L);
            }
        }
    }
}

std::vector<double> compute_region_density(const std::vector<std::vector<double>> &positions) {
    std::vector<double> rho(num_regions, 0.0);
    for (const auto &pos : positions) {
        int x_region = int(pos[0] / (L / grid_size));
        int y_region = int(pos[1] / (L / grid_size));
        if (x_region < 0) x_region = 0;
        if (x_region >= grid_size) x_region = grid_size - 1;
        if (y_region < 0) y_region = 0;
        if (y_region >= grid_size) y_region = grid_size - 1;
        int region = x_region + grid_size * y_region;
        rho[region] += 1.0 / N;
    }
    return rho;
}

void compute_cross_correlation(const std::vector<std::vector<double>> &density_history,
                               std::vector<std::vector<double>> &cross_correlation,
                               int lag_periods) {
    cross_correlation.assign(num_regions, std::vector<double>(num_regions, 0.0));
    if (density_history.size() <= static_cast<size_t>(lag_periods)) return;

    int count_samples = 0;
    #pragma omp parallel for reduction(+:count_samples)
    for (size_t t = 0; t < density_history.size() - lag_periods; ++t) {
        for (int i = 0; i < num_regions; ++i) {
            for (int j = 0; j < num_regions; ++j) {
                #pragma omp atomic
                cross_correlation[i][j] += density_history[t][i] * density_history[t + lag_periods][j];
            }
        }
        count_samples++;
    }
    if (count_samples > 0) {
        for (int i = 0; i < num_regions; ++i) {
            for (int j = 0; j < num_regions; ++j) {
                cross_correlation[i][j] /= count_samples;
            }
        }
    }
}

double compute_sigma_cg(const std::vector<std::vector<double>> &cross_correlation,
                        int lag_periods) {
    double sigma_cg = 0.0;
    double tau = lag_periods * T;
    for (int i = 0; i < num_regions; ++i) {
        for (int j = 0; j < num_regions; ++j) {
            if (i != j && cross_correlation[i][j] > 0 && cross_correlation[j][i] > 0) {
                double ratio = cross_correlation[i][j] / cross_correlation[j][i];
                sigma_cg += (cross_correlation[i][j] - cross_correlation[j][i]) * std::log(ratio);
            }
        }
    }
    // Returns the one-period estimator (T/tau) sigma_est_[0,tau].
    return 0.5 * sigma_cg * (T / tau);
}

void simulate(double E_field, double period,
              std::vector<std::vector<double>> &density_history,
              std::vector<std::vector<double>> &positions,
              const std::vector<double> &sigmas,
              const std::vector<double> &diffusion_coefficients,
              const std::vector<double> &charges,
              double &real_entropy_production) {
    density_history.clear();
    const int steps_per_period = static_cast<int>(std::llround(period / dt));
    const int phase_offset_steps = static_cast<int>(std::llround(sampling_phase * period / dt));
    const int wait_to_phase = (phase_offset_steps - (equilibration_steps % steps_per_period) + steps_per_period) % steps_per_period;
    const int production_start_step = equilibration_steps + wait_to_phase;
    const int total_steps = production_start_step + production_steps;
    density_history.reserve(production_steps / steps_per_period + 2);
    real_entropy_production = 0.0;

    std::vector<std::vector<double>> forces(N, std::vector<double>(2, 0.0));
    std::vector<std::vector<double>> old_positions(N, std::vector<double>(2, 0.0));
    const int display_interval = 10000;

    for (int step = 0; step < total_steps; ++step) {
        double t = step * dt;

        // Store only phase-locked stroboscopic configurations t=t0+kT.
        if (step >= production_start_step && ((step - production_start_step) % steps_per_period == 0)) {
            density_history.push_back(compute_region_density(positions));
        }

        if (step >= production_start_step) old_positions = positions;

        for (auto &f : forces) f[0] = f[1] = 0.0;
        compute_wca_force(positions, sigmas, forces);
        compute_spring_force(positions, forces);
        compute_electric_force(charges, forces, E_field, t, period);

        bool ok = true;
        for (auto &f : forces) {
            if (!is_valid_number(f[0]) || !is_valid_number(f[1])) { ok = false; break; }
        }
        if (!ok) { std::cerr << "Bad force at step " << step << "\n"; continue; }

        langevin_step(positions, forces, diffusion_coefficients);

        ok = true;
        for (auto &p : positions) {
            if (!is_valid_number(p[0]) || !is_valid_number(p[1])) { ok = false; break; }
        }
        if (!ok) { std::cerr << "Bad pos at step " << step << "\n"; continue; }

        if (step >= production_start_step) {
            double inst_p = 0.0;
            double E_mid = electric_field_value(E_field, t + 0.5 * dt, period);
            for (int i = 0; i < N; ++i) {
                double dx = positions[i][0] - old_positions[i][0];
                if      (dx >  L / 2) dx -= L;
                else if (dx < -L / 2) dx += L;
                double v_mid = dx / dt;
                inst_p += (charges[i] * E_mid) * v_mid;
            }
            real_entropy_production += inst_p;
        }

        if (step % display_interval == 0) {
            if (step < production_start_step) {
                double pct = 100.0 * step / production_start_step;
                std::cout << "E=" << E_field << " EQ/phase " << pct << "%\r";
            } else {
                int p = step - production_start_step;
                double pct = 100.0 * p / production_steps;
                std::cout << "E=" << E_field << " PR " << pct << "%\r";
            }
            std::cout.flush();
        }
    }

    // Average entropy production per period.  Since T=1 in the simulations,
    // this is numerically also the steady period-averaged EPR.
    real_entropy_production /= (production_steps * (k_BT / period));
    std::cout << "\nFinished sim E=" << E_field
              << ", stroboscopic samples=" << density_history.size() << "\n";
}

int main() {
    omp_set_num_threads(1);
    std::ios::sync_with_stdio(false);
    std::cout.setf(std::ios::unitbuf);

    std::vector<int> N_values = {1, 2, 5, 10, 20, 50, 60, 80, 100};
    const int ensemble_size = 5;

    // Integer-period lag times used for the main-text estimator.
    const std::vector<int> ell_list = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 22, 25, 30, 40};

    for (int n_val : N_values) {
        N = n_val;
        std::vector<std::vector<double>> base_positions;
        std::vector<double> sigmas, diffusion_coeffs, charges;
        initialize_particles(base_positions, sigmas, diffusion_coeffs, charges);

        for (int sim = 0; sim < ensemble_size; ++sim) {
            std::vector<std::vector<double>> positions = base_positions;
            auto initial_positions = positions;

            std::string fname = "results_phase_locked_N" + std::to_string(N)
                              + "_sim" + std::to_string(sim + 1) + ".csv";
            std::ofstream fout(fname);
            fout << "E,sigma_real";
            for (int ell : ell_list) fout << ",sigma_cg_ell" << ell;
            fout << "\n";

            for (double E : E_field_steps) {
                positions = initial_positions;
                std::vector<std::vector<double>> density_history;
                double sigma_real = 0.0;
                simulate(E, T, density_history, positions,
                         sigmas, diffusion_coeffs, charges, sigma_real);

                std::vector<double> scg_vals;
                scg_vals.reserve(ell_list.size());
                for (int ell : ell_list) {
                    std::vector<std::vector<double>> C;
                    compute_cross_correlation(density_history, C, ell);
                    double scg = compute_sigma_cg(C, ell);
                    scg_vals.push_back(scg);
                }

                fout << E << "," << sigma_real;
                for (double scg : scg_vals) fout << "," << scg;
                fout << "\n";
                fout.flush();
            }
            fout.close();
            std::cout << "Finished N=" << N << " sim#" << sim + 1 << "\n";
        }
    }

    std::cout << "All simulations completed.\n";
    return 0;
}
