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
const double T = 1.0; //period of the electric force
const int equilibration_steps = 3000000;
const int production_steps = 7200000;
const double friction_coeff = 1.0;
//const int delta_t_steps = 10;  // cross correlation time
const int grid_size = 5;  // 每个方向上的格子数，10×10=100个格子  
const int num_regions = grid_size * grid_size;  // 总区域数  
const std::vector<double> E_field_steps = {0.0, 0.1, 0.2, 0.4, 0.5, 0.8, 1.0};//
const std::vector<double> sigma_values = {0.01, 0.02, 0.03};
//const std::vector<double> D_values = {0.95, 1.0, 1.05};
const std::vector<double> charge_values = {0.9, 1.0, 1.1};
// 模拟里每个生产步都存一次 positions_history
const int obs_interval = 1;
//const std::vector<int> delta_t_list = {
//    100,  200,  500,  1000, 1500, 1800,
 //   2000, 5000, 9000, 10000, 15000, 18000};
// 用于环缓冲
const int max_delta = 18000;

// 在常量定义后添加  
inline bool is_valid_number(double x) {  
    return !std::isnan(x) && !std::isinf(x) && std::abs(x) < 1e6;  
}

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> uniform_dist(0.0, L);
std::uniform_int_distribution<> sigma_dist(0, sigma_values.size() - 1);
//std::uniform_int_distribution<> D_dist(0, D_values.size() - 1);
std::uniform_int_distribution<> charge_dist(0, charge_values.size() - 1);
std::normal_distribution<> normal_dist(0.0, 1.0);

// Utility functions
double minimum_image_distance(double delta, double box_length) {
    delta -= box_length * std::round(delta / box_length);
    return delta;
}

// Particle initialization
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

                if (r2 < pow(2.0, 1.0 / 3.0) * sigma * sigma) {  
                    double r2_inv = 1.0 / r2;  
                    double r6_inv = pow(r2_inv, 3);  
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

                if (r2 < pow(2.0, 1.0 / 3.0) * sigma * sigma) {  
                    double r2_inv = 1.0 / r2;  
                    double r6_inv = pow(r2_inv, 3);  
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
                double r = sqrt(dx * dx + dy * dy);  

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
                double r = sqrt(dx * dx + dy * dy);  

                if (r > 1e-10) {  
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
                            double E, double t, double T) {
    double E_t = E + E * std::sin(2.0 * M_PI * t / T); // 计算时间依赖的电场
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
            if (is_valid_number(force)) {
                forces[i][0] += force;
            }
        }
    }
}



void langevin_step(std::vector<std::vector<double>> &positions,  
                   const std::vector<std::vector<double>> &forces,  
                   const std::vector<double> &diffusion_coefficients) {  
    #pragma omp parallel  
    {  
        // improved random number generator
        std::mt19937 local_gen(rd() + omp_get_thread_num() * 1000);  
        std::normal_distribution<double> local_normal(0.0, 1.0);  
 

        #pragma omp for  
        for (int i = 0; i < N; ++i) {  
            // 限制噪声项的大小  
            double noise_x = std::min(std::max(  
                sqrt(2.0 * diffusion_coefficients[i] * dt / friction_coeff) * local_normal(local_gen),  
                -0.1), 0.1);  
            double noise_y = std::min(std::max(  
                sqrt(2.0 * diffusion_coefficients[i] * dt / friction_coeff) * local_normal(local_gen),  
                -0.1), 0.1);  

            // 限制力导致的位移  
            double dx = std::min(std::max((diffusion_coefficients[i] * forces[i][0]) * dt, -1.0), 1.0);  
            double dy = std::min(std::max((diffusion_coefficients[i] * forces[i][1]) * dt, -1.0), 1.0);  

            if (is_valid_number(dx + noise_x) && is_valid_number(dy + noise_y)) {  
                positions[i][0] += dx + noise_x;  
                positions[i][1] += dy + noise_y;  

                positions[i][0] = fmod(positions[i][0] + L, L);  
                positions[i][1] = fmod(positions[i][1] + L, L);  
            }  
        }  
    }  
}


void compute_cross_correlation(const std::vector<std::vector<std::vector<double>>> &positions_history,  
                             std::vector<std::vector<double>> &cross_correlation,  
                             int delta_t_steps) {  
    // 初始化相关矩阵为num_regions × num_regions  
    cross_correlation.assign(num_regions, std::vector<double>(num_regions, 0.0));  
    int count_samples = 0;  

    #pragma omp parallel for reduction(+:count_samples)  
    for (size_t t = 0; t < positions_history.size() - delta_t_steps; ++t) {  
        std::vector<double> region_density_t(num_regions, 0.0);  
        std::vector<double> region_density_tp(num_regions, 0.0);  

        // 计算t时刻的密度  
        for (const auto &pos : positions_history[t]) {  
            // 使用grid_size进行区域划分  
            int x_region = int(pos[0] / (L / grid_size));  
            int y_region = int(pos[1] / (L / grid_size));  
            int region = x_region + grid_size * y_region;  // 修改区域索引计算  
            if (region >= 0 && region < num_regions) {  // 添加边界检查  
                region_density_t[region] += 1.0/N;  
            }  
        }  

        // 计算t+delta_t时刻的密度  
        for (const auto &pos : positions_history[t + delta_t_steps]) {  
            int x_region = int(pos[0] / (L / grid_size));  
            int y_region = int(pos[1] / (L / grid_size));  
            int region = x_region + grid_size * y_region;  
            if (region >= 0 && region < num_regions) {  
                region_density_tp[region] += 1.0/N;  
            }  
        }  

        // 计算密度相关函数  
        for (int i = 0; i < num_regions; ++i) {  
            for (int j = 0; j < num_regions; ++j) {  
                cross_correlation[i][j] += region_density_t[i] * region_density_tp[j];  
            }  
        }  
        count_samples++;  
    }  

    // 归一化  
    if (count_samples > 0) {  
        for (int i = 0; i < num_regions; ++i) {  
            for (int j = 0; j < num_regions; ++j) {  
                cross_correlation[i][j] /= count_samples;  
            }  
        }  
    }  
}  

double compute_sigma_cg(const std::vector<std::vector<double>> &cross_correlation,  
                       int delta_t_steps,  
                       double dt) {  
    double sigma_cg = 0.0;  
    double delta_t = delta_t_steps * dt;  

    // 遍历所有区域对  
    for (int i = 0; i < num_regions; ++i) {  
        for (int j = 0; j < num_regions; ++j) {  
            if (i != j && cross_correlation[i][j] > 0 && cross_correlation[j][i] > 0) {  
                double ratio = cross_correlation[i][j] / cross_correlation[j][i];  
                sigma_cg += (cross_correlation[i][j] - cross_correlation[j][i]) * log(ratio);  
            }  
        }  
    }  
    return 0.5 * sigma_cg * (T/delta_t);  
}

// ---------------------------------------------------
// simulate：完全恢复原来逻辑，按 dt 分辨率存储 positions
// ---------------------------------------------------
void simulate(double E_field, double T,
              std::vector<std::vector<std::vector<double>>> &positions_history,
              std::vector<std::vector<std::vector<double>>> &forces_history,
              std::vector<std::vector<double>> &positions,
              const std::vector<double> &sigmas,
              const std::vector<double> &diffusion_coefficients,
              const std::vector<double> &charges,
              double &real_entropy_production)
{
    positions_history.clear();
    positions_history.reserve(production_steps);
    forces_history.clear();
    forces_history.reserve(production_steps);
    real_entropy_production = 0.0;

    std::vector<std::vector<double>> forces(N, std::vector<double>(2,0.0));
    std::vector<std::vector<double>> old_positions(N, std::vector<double>(2,0.0));

    const int display_interval = 10000;
    int total_steps = equilibration_steps + production_steps;

    for (int step = 0; step < total_steps; ++step) {
        double t = step * dt;

        // Stratonovich 用的旧位置
        if (step >= equilibration_steps) old_positions = positions;

        // 1. 计算力
        for (auto &f : forces) f[0]=f[1]=0.0;
        compute_wca_force(positions, sigmas, forces);
        compute_spring_force(positions, forces);
        compute_electric_force(charges, forces, E_field, t, T);

        // 力合法性
        bool ok = true;
        for (auto &f : forces) {
            if (!is_valid_number(f[0]) || !is_valid_number(f[1])) { ok=false; break; }
        }
        if (!ok) { std::cerr<<"Bad force at step "<<step<<"\n"; continue; }

        // 2. Langevin 步
        langevin_step(positions, forces, diffusion_coefficients);

        // 位置合法性
        ok = true;
        for (auto &p : positions) {
            if (!is_valid_number(p[0]) || !is_valid_number(p[1])) { ok=false; break; }
        }
        if (!ok) { std::cerr<<"Bad pos at step "<<step<<"\n"; continue; }

        // 3. 生产阶段处理
        if (step >= equilibration_steps) {
            // 3.1 累加真实熵
            double inst_p = 0;
            for (int i = 0; i < N; ++i) {
                double dx = positions[i][0] - old_positions[i][0];
                if      (dx >  L/2) dx -= L;
                else if (dx < -L/2) dx += L;
                double v_mid = dx/dt;
                inst_p += (charges[i]*E_field)*v_mid;
            }
            real_entropy_production += inst_p;

            // 3.2 存历史
            positions_history.push_back(positions);
            forces_history   .push_back(forces);
        }

        // 4. 进度显示
        if (step % display_interval == 0) {
            if (step < equilibration_steps) {
                double pct = 100.0*step/equilibration_steps;
                std::cout<<"E="<<E_field<<" EQ "<<pct<<"%\r";
            } else {
                int p = step - equilibration_steps;
                double pct = 100.0*p/production_steps;
                std::cout<<"E="<<E_field<<" PR "<<pct<<"%\r";
            }
            std::cout.flush();
        }
    }

    real_entropy_production /= (production_steps*(k_BT/T));
    std::cout<<"\nFinished sim E="<<E_field<<"\n";
}


int main() {
    omp_set_num_threads(1);
    std::ios::sync_with_stdio(false);
    std::cout.setf(std::ios::unitbuf);

    // 要跑的 N 值和系综次数
    std::vector<int> N_values = {1, 2, 5, 10, 20, 50, 60, 80, 100};
    const int ensemble_size = 5;

    // Δt 列表（仿真步数）
    const std::vector<int> delta_t_list = {
        1000, 1500, 1800,
        2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000, 15000, 18000, 20000, 22000, 25000, 30000, 32000, 35000, 40000
    };

    // 遍历不同粒子数
    for (int n_val : N_values) {
        N = n_val;  // 全局粒子数
        std::vector<std::vector<double>> base_positions;
        std::vector<double> sigmas, diffusion_coeffs, charges;
        initialize_particles(base_positions, sigmas, diffusion_coeffs, charges);
        // 每个 N 做 ensemble_size 次
        for (int sim = 0; sim < ensemble_size; ++sim) {
            // 初始化体系
            std::vector<std::vector<double>> positions = base_positions;
            auto initial_positions = positions;

            // 打开输出 CSV
            std::string fname = "results_N" 
                              + std::to_string(N)
                              + "_sim" + std::to_string(sim+1)
                              + ".csv";
            std::ofstream fout(fname);

            // 写表头
            fout << "E,sigma_real";
            for (int dt_step : delta_t_list) {
                fout << ",sigma_cg_" << dt_step;
            }
            fout << "\n";

            // 针对每个电场强度跑模拟 + 后处理
            for (double E : E_field_steps) {
                // 恢复初始配置
                positions = initial_positions;

                // 运行模拟，得到 positions_history 和 sigma_real
                std::vector<std::vector<std::vector<double>>> positions_history;
                std::vector<std::vector<std::vector<double>>> forces_history;
                double sigma_real = 0.0;
                simulate(E, T,
                         positions_history,
                         forces_history,
                         positions,
                         sigmas,
                         diffusion_coeffs,
                         charges,
                         sigma_real);

                // 计算各 Δt 下的 sigma_cg
                std::vector<double> scg_vals;
                scg_vals.reserve(delta_t_list.size());
                for (int dt_step : delta_t_list) {
                    // 直接用 dt_step 作为延迟步数
                    std::vector<std::vector<double>> C;
                    compute_cross_correlation(
                        positions_history, C, dt_step);
                    double scg = compute_sigma_cg(
                        C, dt_step, dt);
                    scg_vals.push_back(scg);
                }

                // 一行写入：E, sigma_real, 然后所有 sigma_cg
                fout << E << "," << sigma_real;
                for (double scg : scg_vals) {
                    fout << "," << scg;
                }
                fout << "\n";
                fout.flush();
            }

            fout.close();
            std::cout << "Finished N=" << N 
                      << " sim#" << sim+1 << "\n";
        }
    }

    std::cout << "All simulations completed.\n";
    return 0;
}


/*
void simulate(double E_field,double T,  
              std::vector<std::vector<std::vector<double>>> &positions_history,  
              std::vector<std::vector<std::vector<double>>> &forces_history,  
              std::vector<std::vector<double>> &positions,  
              const std::vector<double> &sigmas,  
              const std::vector<double> &diffusion_coefficients,  
              const std::vector<double> &charges) {  
    try {  
        positions_history.clear();  
        forces_history.clear();  

        std::vector<std::vector<double>> forces(N, std::vector<double>(2, 0.0));  
        
        // 设置更小的显示间隔  
        const int display_interval = 10000;  
        
        std::cout << "Starting simulation for E = " << E_field << std::endl;  

        for (int step = 0; step < equilibration_steps + production_steps; ++step) { 
            double t = step * dt; 
            // 重置力  
            for (auto &force : forces) {  
                force[0] = force[1] = 0.0;  
            }  

            compute_wca_force(positions, sigmas, forces);  
            compute_spring_force(positions, forces);  
            compute_electric_force(charges, forces, E_field, t, T);  

            // 检查力是否有效  
            bool forces_valid = true;  
            for (const auto &force : forces) {  
                if (!is_valid_number(force[0]) || !is_valid_number(force[1])) {  
                    forces_valid = false;  
                    break;  
                }  
            }  

            if (!forces_valid) {  
                std::cerr << "Warning: Invalid forces detected at step " << step << std::endl;  
                continue;  
            }  

            langevin_step(positions, forces, diffusion_coefficients);  

            // 检查位置是否有效  
            bool positions_valid = true;  
            for (const auto &pos : positions) {  
                if (!is_valid_number(pos[0]) || !is_valid_number(pos[1])) {  
                    positions_valid = false;  
                    break;  
                }  
            }  

            if (!positions_valid) {  
                std::cerr << "Warning: Invalid positions detected at step " << step << std::endl;  
                continue;  
            }  

            if (step >= equilibration_steps) {  
                positions_history.push_back(positions);  
                forces_history.push_back(forces);  
            }  

            // 修改进度显示逻辑  
            if (step % display_interval == 0) {  
                std::cout << std::fixed << std::setprecision(1);  
                if (step < equilibration_steps) {  
                    double progress = (step * 100.0) / equilibration_steps;  
                    std::cout << "E = " << E_field << " Equilibration: Step " << step   
                              << "/" << equilibration_steps << " ("   
                              << progress << "%)                    \n";  
                } else {  
                    int prod_step = step - equilibration_steps;  
                    double progress = (prod_step * 100.0) / production_steps;  
                    std::cout << "E = " << E_field << " Production: Step "   
                              << prod_step << "/" << production_steps << " ("   
                              << progress << "%)                    \n";  
                }  
                std::cout.flush();  
            }  
        }  
        std::cout << "Completed simulation for E = " << E_field << std::endl;  

    } catch (const std::exception& e) {  
        std::cerr << "Error in simulation at E = " << E_field << ": " << e.what() << std::endl;  
        throw;  
    } catch (...) {  
        std::cerr << "Unknown error in simulation at E = " << E_field << std::endl;  
        throw;  
    }  
}

// Main function
int main() {  
    omp_set_num_threads(1);  // 将线程数限制为 16
    const int NUM_SIMULATIONS = 30;  // 模拟次数  
    
    std::ios_base::sync_with_stdio(false);  
    std::cout.setf(std::ios::unitbuf);  
    
    // 只需要一个snapshot文件  
    std::ofstream snapshot_file("snapshot_E_0.5.csv");  
    snapshot_file << "x,y,charge,sigma\n";  
    
    // 初始化粒子  
    std::vector<std::vector<double>> positions;  
    std::vector<double> sigmas, diffusion_coefficients, charges;  
    initialize_particles(positions, sigmas, diffusion_coefficients, charges);  

    // 保存初始位置的深拷贝  
    std::vector<std::vector<double>> initial_positions = positions;  

    // 逐个完成每次独立模拟  
    for (int sim = 0; sim < NUM_SIMULATIONS; ++sim) { 
        int steps_array[NUM_SIMULATIONS] = {  
            2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000,
            500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
            15000, 15000, 15000, 18000, 18000, 18000, 10000, 10000, 10000, 10000};  
        int delta_t_steps = steps_array[sim];
        std::cout << "Starting simulation " << sim + 1 << " of " << NUM_SIMULATIONS << std::endl;  

        // 为当前模拟创建结果文件  
        std::string filename = "sigma_vs_E_sim_" + std::to_string(sim + 1) + ".csv";  
        std::ofstream sigma_file(filename);  
        sigma_file << "E,sigma_cg,real_entropy_production\n";  

        // 对每个电场强度进行模拟  
        for (double E : E_field_steps) {  
            std::cout << "  Processing E = " << E << std::endl;  
            positions = initial_positions;
            // Run simulation  
            std::vector<std::vector<std::vector<double>>> positions_history;  
            std::vector<std::vector<std::vector<double>>> forces_history;  
            simulate(E, T, positions_history, forces_history, positions, sigmas, diffusion_coefficients, charges);  

            // Compute cross-correlation and sigma_cg  
            std::vector<std::vector<double>> cross_correlation;  
            compute_cross_correlation(positions_history, cross_correlation, delta_t_steps);  
            double sigma_cg = compute_sigma_cg(cross_correlation, delta_t_steps, dt);  

            // Compute real entropy production using Stratonovich interpretation  
            double real_entropy_production = 0.0;  
            std::vector<std::vector<double>> total_displacements(N, std::vector<double>(2, 0.0));  

            for (size_t t = 0; t < positions_history.size() - 1; ++t) {  
                double instant_power = 0.0;  
                
                for (int i = 0; i < N; ++i) {  
                    double F_ext = charges[i] * E;  
                    
                    // 计算相邻时间步的位移  
                    double dx = positions_history[t+1][i][0] - positions_history[t][i][0];  
                    
                    // 处理周期性边界条件下的位移  
                    if (dx > L/2) {  
                        dx -= L;  
                    } else if (dx < -L/2) {  
                        dx += L;  
                    }  
                    
                    // 计算中点速度 (Stratonovich interpretation)  
                    double x_mid = 0.5 * (positions_history[t+1][i][0] + positions_history[t][i][0]);  
                    double v_mid = dx / dt;  
                    
                    // 使用中点速度计算功率  
                    instant_power += F_ext * v_mid;  
                }  
                real_entropy_production += instant_power;  
            }  

            // 对时间步数平均，并无量纲化  
            real_entropy_production /= ((positions_history.size() - 1) * k_BT/T);  

            // 将结果写入当前模拟的文件  
            sigma_file << E << "," << sigma_cg << "," << real_entropy_production << "\n";
            
            // 确保立即写入磁盘  
            sigma_file.flush();  

            // 只在第一次模拟且E=0.5时保存snapshot  
            if (sim == 0 && E == 0.5) {  
                for (int i = 0; i < N; ++i) {  
                    int random_charge_idx = charge_dist(gen);  
                    int random_sigma_idx = sigma_dist(gen);  
                    snapshot_file << positions[i][0] << "," << positions[i][1] << ","  
                            << charge_values[random_charge_idx] << ","  
                            << sigma_values[random_sigma_idx] << "\n";  
                }  
                snapshot_file.flush();  
            }  
        }  

        // 完成当前模拟，关闭文件  
        sigma_file.close();  
        std::cout << "Simulation " << sim + 1 << " completed." << std::endl;  
    }  

    // 关闭snapshot文件  
    snapshot_file.close();  

    std::cout << "All simulations completed. Results saved to files.\n";  
    return 0;  
}*/
