/*
MotionPrimitiveListTree
Copyright (C) 2019 Xuning Yang

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <motion_primitive_tree/ListTree.h>
#include <cost/Cost.h>

#include <chrono>
#include <stdexcept>

#include <ros_utils/ParameterUtils.h>

// #include <planning_arch/trajectory/trajectory.h>

namespace cost = planner::cost_function;
namespace stats = stats_utils;
namespace su = sample_utils;
namespace vu = vector_utils;

using Clock = std::chrono::high_resolution_clock;
using NumLimitFloat = std::numeric_limits<float>;
using NumLimitDouble = std::numeric_limits<double>;

namespace planner {

MotionPrimitiveListTree::MotionPrimitiveListTree() {
  tree_initialized_ = false;
}

MotionPrimitiveListTree::~MotionPrimitiveListTree() {

  if (!tree_initialized_) return;

  // publish some debug results at kill time.
  std::cout << "===Summary of MotionPrimitiveListTree===" << std::endl;
  std::cout << "Number of trees created: " << num_trees_created_ << std::endl;

  int stddev_iter;
  int avg_iter = stats::Average(iter_, &stddev_iter);
  std::cout << "Average number of iter per tree: " << avg_iter << " +- " << stddev_iter << std::endl;

  int stddev_all_trajectories_generated;
  int avg_all_trajectories_generated = stats::Average(all_trajectories_generated_, &stddev_all_trajectories_generated);
  std::cout << "Average num of trajectories per tree: " << avg_all_trajectories_generated << " +- " << stddev_all_trajectories_generated << std::endl;

  int stddev_nodes_added;
  int avg_nodes_added = stats::Average(nodes_added_, &stddev_nodes_added);
  std::cout << "Average num of nodes ADDED per tree: " << avg_nodes_added << " +- " << stddev_nodes_added << std::endl;

  int stddev_nodes_rejected;
  int avg_nodes_rejected = stats::Average(nodes_rejected_, &stddev_nodes_rejected);
  std::cout << "Average num of nodes REJECTED per tree: " << avg_nodes_rejected << " +- " << stddev_nodes_rejected << std::endl;

  int stddev_num_nodes_processed;
  int avg_num_nodes_processed = stats::Average(num_nodes_processed_, &stddev_num_nodes_processed);
  std::cout << "Average num of nodes PROCESSED per tree: " << avg_num_nodes_processed << " +- " << stddev_num_nodes_processed << std::endl;

  int stddev_depth;
  int avg_depth = stats::Average(depth_of_tree_, &stddev_depth);
  std::cout << "Average max depth of tree: " << avg_depth << " +- " << stddev_depth << std::endl;

  float stddev_mincost;
  float avg_mincost = stats::Average(min_cost_, &stddev_mincost);
  float max_mincost = vu::Max(min_cost_);
  std::cout << "Average min cost: " << avg_mincost << " +- " << stddev_mincost <<" Max: " << max_mincost << std::endl;

  float stddev_dist_to_goal;
  float avg_dist_to_goal = 0.0;
  float max_dist_to_goal = 0.0;
  if (w_goal_ != 0)
  {
    avg_dist_to_goal = stats::Average(dist_to_goal_, &stddev_dist_to_goal);
    max_dist_to_goal = vu::Max(dist_to_goal_);
    std::cout << "Average dist to goal: " << avg_dist_to_goal << " +- " << stddev_dist_to_goal <<" Max: " << max_dist_to_goal << std::endl;
  }

  float stddev_jerk;
  float avg_jerk = stats::Average(jerk_integral_, &stddev_jerk);
  float max_jerk = vu::Max(jerk_integral_);
  std::cout << "Average jerk integral: " << avg_jerk << " +- " << stddev_jerk <<" Max: " << max_jerk << std::endl;

  if (record_) {
    tree_file_ << "Name, Avg, Stddev, Min, Max" << std::endl;

    tree_file_<< "Number of trees created," << num_trees_created_ << ",NaN, NaN,NaN"<< std::endl;

    tree_file_ << "Num. of iter per tree," << avg_iter << "," << stddev_iter << ",NaN,NaN" << std::endl;

    tree_file_ << "Num. of trajectories per tree," << avg_all_trajectories_generated << "," << stddev_all_trajectories_generated << ",NaN,NaN" << std::endl;

    tree_file_ << "Num. of nodes ADDED per tree," << avg_nodes_added << "," <<  stddev_nodes_added << ",NaN,NaN" <<  std::endl;

    tree_file_ << "Num. of nodes REJECTED per tree," << avg_nodes_rejected << "," << stddev_nodes_rejected << ",NaN,NaN" << std::endl;

    tree_file_ << "Num. of nodes PROCESSED per tree," << avg_num_nodes_processed << "," << stddev_num_nodes_processed << ",NaN,NaN" << std::endl;

    tree_file_ << "Max depth of tree," << avg_depth << "," << stddev_depth << ",NaN,NaN" << std::endl;

    tree_file_ << "Min cost," << avg_mincost << "," << stddev_mincost << ",NaN," << max_mincost << std::endl;

    if (w_goal_ != 0)
    {
      tree_file_ << "Dist to goal," << avg_dist_to_goal << "," << stddev_dist_to_goal <<",NaN," << max_dist_to_goal << std::endl;
    }

    tree_file_ << "Jerk integral," << avg_jerk << "," << stddev_jerk <<",NaN," << max_jerk << std::endl;

    tree_file_.close();
  }

  std::cout << "======================================" << std::endl;


}


void MotionPrimitiveListTree::initialize(const ros::NodeHandle& n,
    const std::vector<double>& x_vel,
    const std::vector<double>& omega_vel,
    const std::vector<double>& z_vel,
    const std::vector<double>& duration)
{

  // this initialization function is for when no collision checker is provided. in which case, it will create and own it.
  auto collision_checker = std::make_shared<CollisionChecker>();
  if (!collision_checker->initialize(n))
  {
    ROS_ERROR("[MotionPrimitiveListTree] unable to initialize collision checker, exiting!");
    return;
  }

  // then call the rest of the initialization function.
  initialize(n, x_vel, omega_vel, z_vel, duration, collision_checker);

  return;
}


void MotionPrimitiveListTree::initialize(const ros::NodeHandle& n,
    const std::vector<double>& x_vel,
    const std::vector<double>& omega_vel,
    const std::vector<double>& z_vel,
    const std::vector<double>& duration,
    const std::shared_ptr<CollisionChecker>& collision_checker)
{
  std::cout << "[ListTree] initializing tree.." << std::endl;

  if (x_vel.size() == 0 || omega_vel.size() == 0 ||
      z_vel.size() == 0 || duration.size() == 0 )
  {
    throw std::invalid_argument("[ListTree] input range needs to be larger than 1.");
  }

  max_duration_ = duration.back();
  max_xvel_ = x_vel.back();
  max_zvel_ = z_vel.back();
  max_omega_ = omega_vel.back();

  // Initialize the static library of primitives.
  for (size_t i = 0; i < x_vel.size(); i++) {
    for (size_t j = 0; j < omega_vel.size(); j++) {
      for (size_t k = 0; k < z_vel.size(); k++) {
        for (size_t l = 0; l < duration.size(); l++) {
          static_library_.emplace_back(Eigen::Vector4d(x_vel[i], omega_vel[j], z_vel[k], 0.0), duration[l]);
        } // end l
      } //end k
    } // end j
  } // end i

  // Add root node to tree.
  leaves_.push_back(0);

  // Initialize multithreading
  param_utils::get("active_replan/num_threads", num_threads_, 4);

  // Initiailize parameters
  param_utils::get("active_replan/tree_size", tree_size_, 150);

  // cost bound parameters
  param_utils::get("active_replan/cost_bound_enabled", cost_bound_enabled_, false);
  param_utils::get("active_replan/percentile", percentile_, (float)0.3);


  // sampling parameters
  param_utils::get("active_replan/sample_batch_size", sample_batch_size_, 1);
  param_utils::get("active_replan/elite_set_size", elite_set_size_, 15);
  param_utils::get("active_replan/softmax_on", softmax_enabled_, false);
  param_utils::get("active_replan/beta", beta_, (float)1.0);

  // Set parameters
  param_utils::get("active_replan/global_goal_x", goal_(0), 5.0);
  param_utils::get("active_replan/global_goal_y", goal_(1), 0.0);
  param_utils::get("active_replan/global_goal_z", goal_(2), 1.0);

  // weights for various cost functions; setting them to zero turns it off.
  param_utils::get("active_replan/w_smooth", w_smooth_, (float)1.0);
  param_utils::get("active_replan/w_straightline", w_sl_, (float)1.0);
  param_utils::get("active_replan/w_speed", w_speed_, (float)1.0);
  param_utils::get("active_replan/w_duration", w_duration_, (float)1.0);
  param_utils::get("active_replan/w_length", w_length_, (float)1.0);

  param_utils::get("active_replan/w_direction", w_direction_, (float)0.0);
  param_utils::get("active_replan/w_input", w_input_, (float)0.0);
  param_utils::get("active_replan/w_goal", w_goal_, (float)0.0);
  param_utils::get("active_replan/w_point", w_point_, (float)0.0);
  param_utils::get("active_replan/w_deviation", w_deviation_, (float)0.0);

  // Setup dynamic reconfigure
  ros::NodeHandle dynreconf_node(n, "active_replan");
  cost_function_dynamic_reconfigs_server_ = std::unique_ptr<dynamic_reconfigure::Server<motion_primitive_tree::CostFunctionWeightsConfig>>(new dynamic_reconfigure::Server<motion_primitive_tree::CostFunctionWeightsConfig>(dynreconf_node));
  cost_function_dynamic_reconfigs_server_->setCallback(std::bind(&MotionPrimitiveListTree::costFunctionWeightReconfigureCallback, this, std::placeholders::_1, std::placeholders::_2));

  // Collision checker
  collision_checker_ = collision_checker;
  // check that collision checker is initialized
  if (!collision_checker_->isInitialized())
  {
    ROS_ERROR("[MotionPrimitiveListTree::initialize] collision checker wasn't properly initialized outside of this function, returning false!");
    return;
  }


  // Initialize generator
  auto seed = Clock::now().time_since_epoch().count();
  generator_ = std::mt19937(seed);

  std::cout << "=======================================================" << std::endl;
  std::cout << "\t\tTree Parameters " << std::endl;
  std::cout << "=======================================================" << std::endl;
  std::cout << "# THREADS: \t\t\t" << num_threads_ << "\n---" << std::endl;

  std::cout << "Tree size: \t\t\t" << tree_size_ <<  " nodes" << std::endl;

  std::cout << "Sample batch size per iter: \t" << sample_batch_size_ << std::endl;
  std::cout << "Sampling from K top elements: \t" << elite_set_size_ << std::endl;
  std::string softmax = (softmax_enabled_) ? "ON" : "OFF";
  std::cout << "Softmax on weights:\t\t" << softmax << std::endl;
  if (softmax_enabled_) std::cout << "\tbeta:\t\t\t" << beta_ << std::endl;
  std::string cost_bound = (cost_bound_enabled_) ? "ON" : "OFF";
  std::cout << "Cost bound enabled:\t\t" << cost_bound << std::endl;
  if (cost_bound_enabled_) std::cout << "\tpercentile cutoff:\t\t" << percentile_ << std::endl;
  std::cout << "---" << std::endl;
  std::cout << "Weights (Dynamic reconfigure available):" << std::endl;
  std::cout << "\tsmoothness:\t\t" << w_smooth_ << std::endl;
  std::cout << "\tstraightline:\t\t" << w_sl_ << std::endl;
  std::cout << "\tspeed:\t\t\t" << w_speed_ << std::endl;
  std::cout << "\tduration:\t\t" << w_duration_ << std::endl;
  std::cout << "\tlength:\t\t\t" << w_length_ << std::endl;
  std::cout << "State space dependent weights: " << std::endl;
  std::cout << "\tdirection:\t\t" << w_direction_ << std::endl;
  std::cout << "\tinput:\t\t\t" << w_input_ << std::endl;
  std::cout << "\tgoal:\t\t\t" << w_goal_ << std::endl;
  std::cout << "\tpoints along a traj to goal:\t" << w_point_ << std::endl;
  std::cout << "---" << std::endl;
  std::cout << "static library initialized with size " << x_vel.size() << " x " << omega_vel.size()  << " x " <<  z_vel.size()  << " x " << duration.size()  << ": " << static_library_.size() << std::endl;
  std::cout << "=======================================================" << std::endl;

  return;
}

void MotionPrimitiveListTree::setUpFileWriting(const std::string& saveto_directory)
{
  record_ = true;

  // tree file
  std::string tree_filename = saveto_directory + std::string("/tree.csv");
  tree_file_.open(tree_filename.c_str());

  // Record all trajectories generated
  std::string pos_filename = saveto_directory + std::string("/all_pos.csv");
  pos_file_.open(pos_filename.c_str());

  std::string vel_filename = saveto_directory + std::string("/all_vel.csv");
  vel_file_.open(vel_filename.c_str());

  std::string acc_filename = saveto_directory + std::string("/all_acc.csv");
  acc_file_.open(acc_filename.c_str());

  std::string jerk_filename = saveto_directory + std::string("/all_jerk.csv");
  jerk_file_.open(jerk_filename.c_str());

  collision_checker_->setUpFileWriting(saveto_directory);
}


void MotionPrimitiveListTree::clear()
{
  tree_.clear();
  leaves_.clear();
  leaves_.push_back(0);
  sample_set_.clear();
  cost_.clear();
  tree_initialized_ = false;
  return;
}

void MotionPrimitiveListTree::costFunctionWeightReconfigureCallback(motion_primitive_tree::CostFunctionWeightsConfig &cfg, uint32_t levels)
{
  ROS_INFO("[MotionPrimitiveListTree::DynamicReconfigure] Weights reconfigured.");
  // Figure out which ones changed.
  if (std::abs(w_smooth_-cfg.w_smooth) > NumLimitFloat::min()) {
    ROS_INFO("  w_smoothness: %.1f --> %.1f",w_smooth_, cfg.w_smooth);
  }
  if (std::abs(w_sl_-cfg.w_straightline) > NumLimitFloat::min()) {
    ROS_INFO("  w_straightline: %.1f --> %.1f",w_sl_, cfg.w_straightline);
  }
  if (std::abs(w_duration_-cfg.w_duration) > NumLimitFloat::min()) {
    ROS_INFO("  w_duration: %.1f --> %.1f",w_duration_, cfg.w_duration);
  }
  if (std::abs(w_speed_-cfg.w_speed) > NumLimitFloat::min()) {
    ROS_INFO("  w_speed: %.1f --> %.1f",w_speed_, cfg.w_speed);
  }
  if (std::abs(w_length_-cfg.w_length) > NumLimitFloat::min()) {
    ROS_INFO("  w_length: %.1f --> %.1f",w_length_, cfg.w_length);
  }
  if (std::abs(w_direction_-cfg.w_direction) > NumLimitFloat::min()) {
    ROS_INFO("  w_direction: %.1f --> %.1f",w_direction_, cfg.w_direction);
  }
  if (std::abs(w_input_-cfg.w_input) > NumLimitFloat::min()) {
    ROS_INFO("  w_input: %.1f --> %.1f", w_input_, cfg.w_input);
  }
  if (std::abs(w_goal_-cfg.w_goal) > NumLimitFloat::min()) {
    ROS_INFO("  w_goal: %.1f --> %.1f", w_goal_, cfg.w_goal);
  }
  if (std::abs(w_point_ - cfg.w_point) > NumLimitFloat::min()) {
    ROS_INFO("  w_point: %.1f --> %.1f", w_point_, cfg.w_point);
  }
  if (std::abs(w_deviation_ - cfg.w_deviation) > NumLimitFloat::min()) {
    ROS_INFO("  w_deviation: %.1f --> %.1f", w_deviation_, cfg.w_deviation);
  }

  if (tree_size_ != cfg.tree_size) {
    ROS_INFO("  Tree Size: %d --> %d", tree_size_, cfg.tree_size);
  }
  if (sample_batch_size_ != cfg.sample_batch_size) {
    ROS_INFO("  Sample batch size: %d --> %d", sample_batch_size_, cfg.sample_batch_size);
  }
  if (elite_set_size_ != cfg.elite_set_size) {
    ROS_INFO("  Top K elements: %d --> %d", elite_set_size_, cfg.elite_set_size);
  }
  if (softmax_enabled_ != cfg.softmax_on) {
    ROS_INFO("  Softmax toggle: %s --> %s", softmax_enabled_?"ON":"OFF", cfg.softmax_on?"ON":"OFF");
  }
  if (std::abs(beta_ - cfg.beta) > NumLimitFloat::min()) {
    ROS_INFO("  Beta: %.1f --> %.1f", beta_, cfg.beta);
  }

  auto new_goal = Eigen::Vector3d(cfg.global_goal_x, cfg.global_goal_y, cfg.global_goal_z);
  if ((goal_ - new_goal).norm() > NumLimitDouble::min()) {
    ROS_INFO("  Goal changed: (%.1f, %.1f, %.1f) --> (%.1f, %.1f, %.1f)", goal_(0), goal_(1), goal_(2), new_goal(0), new_goal(1), new_goal(2));
  }

  auto new_input = Eigen::Vector4d(cfg.vx, cfg.omega, cfg.vz, 0);
  if ((new_input - test_input_).norm() > NumLimitDouble::min()) {
    ROS_INFO("  Input changed: (%.1f, %.1f, %.1f) --> (%.1f, %.1f, %.1f)", test_input_(0), test_input_(1), test_input_(2), new_input(0), new_input(1), new_input(2));
  }


  w_smooth_ = cfg.w_smooth;
  w_sl_ = cfg.w_straightline;
  w_duration_ = cfg.w_duration;
  w_speed_= cfg.w_speed;
  w_direction_ = cfg.w_direction;
  w_input_ = cfg.w_input;
  w_goal_ = cfg.w_goal;
  w_point_ = cfg.w_point;
  w_deviation_ = cfg.w_deviation;

  tree_size_ = cfg.tree_size;
  sample_batch_size_ = cfg.sample_batch_size;
  elite_set_size_ = cfg.elite_set_size;
  softmax_enabled_ = cfg.softmax_on;
  beta_ = cfg.beta;

  goal_ = new_goal;
  test_input_ = new_input;

}

bool MotionPrimitiveListTree::buildTreeToDepth(unsigned int depth)
{
  if (!tree_initialized_) {
    std::cout << "[MotionPrimitiveListTree] Tree not initialized; aborting." << std::endl;
    return false;
  }
  std::cout << "building tree...." << std::endl;

  auto t_start = Clock::now();
  unsigned int d = 0;
  while (!leaves_.empty()) {
    // get the leaf node
    unsigned int parent = leaves_.front();

    d = std::get<2>(tree_[parent]) + 1;
    if (d > depth) break;

    for (size_t i = 0; i < static_library_.size(); i++) {
      // Create a branch from parent node 'parent' to current branch
      tree_.emplace_back(parent, i, d, 0.0);
      // Add the index of the node just inserted into the list.
      leaves_.push_back(tree_.size() - 1);

    }
    leaves_.pop_front();

  }
  std::chrono::duration<double> t_buildtree = Clock::now() - t_start;
  std::cout << "Time to build tree: " << t_buildtree.count() << " s (" << t_buildtree.count() / tree_.size() << " s per node)" << std::endl;

  std::cout << "Number of nodes: " << tree_.size() << std::endl;
  std::cout << "Number of leaves: " << leaves_.size() << std::endl;
  std::cout << "size of static tree: " << static_library_.size() << std::endl;


  return true;
}


std::vector<MotionPrimitiveListTree::Sequence> MotionPrimitiveListTree::getAllLeafSequences()
{
  std::vector<Sequence> sequences;
  if (!tree_initialized_ ) {
    std::cout << "[MotionPrimitiveListTree::getAllLeafSequences] Tree not initialized; aborting." << std::endl;
    return sequences;
  }

  auto t_start = Clock::now();

  for (size_t i = 0; i < leaves_.size(); i++) {
    // get leaf node
    auto current_node = leaves_[i];

    Sequence sequence = getSequence(current_node);
    if (sequence.size() != 0) {
      sequences.push_back(sequence);
    }
  }

  std::chrono::duration<double> t_fetchtraj = Clock::now() - t_start;
  std::cout << "Time to fetch all leaf trajectories (" << sequences.size() << "): " << t_fetchtraj.count() << " s (" << t_fetchtraj.count() / sequences.size() << "s per traj) " << std::endl;

  leaf_trajectories_generated_.push_back(sequences.size());

  return sequences;
}

std::vector<MotionPrimitiveListTree::Sequence> MotionPrimitiveListTree::getAllSequences() {
  std::vector<Sequence> sequences;
  if (!tree_initialized_) {
    std::cout << "[MotionPrimitiveListTree::getAllSequences] Tree not initialized; aborting." << std::endl;
    return sequences;
  }

  auto t_start = Clock::now();

  for (size_t i = 0; i < tree_.size(); i++) {
    // get leaf node
    Sequence sequence = getSequence(i);
    if (sequence.size() != 0) {
      sequences.push_back(sequence);
    }
  }

  std::chrono::duration<double> t_fetchtraj = Clock::now() - t_start;
  std::cout << "Time to fetch all trajectories: " << t_fetchtraj.count() << " s (" << t_fetchtraj.count() / sequences.size() << "s per traj) " << std::endl;

  // Record how many trajectories were generated.
  all_trajectories_generated_.push_back(sequences.size());

  // Record the maximum depth saw
  auto maxdepth_comparator = [](Branch left, Branch right)
    { return std::get<2>(left) < std::get<2>(right); };
  Branch max_depth_node = *std::max_element(tree_.begin(), tree_.end(), maxdepth_comparator);
  depth_of_tree_.push_back(std::get<2>(max_depth_node));

  // Find average length of the trajectories

  return sequences;
}

MotionPrimitiveListTree::Sequence MotionPrimitiveListTree::getSequence(unsigned int leaf) {
  // This function returns a sequence associated with a single leaf.
  Sequence sequence;
  auto current_node = leaf;
  auto depth = std::get<2>(tree_[current_node]);
  if (depth == 0) return sequence;

  auto parent = std::get<0>(tree_[current_node]);
  // Get the index of the primitive from the tree
  auto primitive_idx = std::get<1>(tree_[current_node]);
  // retrieve primitive parameters from the static library
  auto primitive = static_library_[primitive_idx];
  // add the primitive parameters to this sequence
  sequence.push_front(primitive);

  std::vector<Branch> node_path_debug;
  node_path_debug.push_back(tree_[current_node]);

  // travel back up the tree recursively.
  while (parent != 0) {

    current_node = parent;

    // Get the index of the primitive from the tree
    auto primitive_idx = std::get<1>(tree_[current_node]);
    parent = std::get<0>(tree_[current_node]);
    node_path_debug.push_back(tree_[current_node]);

    // retrieve primitive parameters from the static library
    auto primitive = static_library_[primitive_idx];

    // add the primitive parameters to this sequence
    sequence.push_front(primitive);

  }

  if (depth != sequence.size()) {
    std::cout << "[ListTree] DEBUG; WARNING: sequence size (" << sequence.size() << ") does not equal to depth ("<< depth << ")" << std::endl;
    printTree();
    printReverseTreeBranch(node_path_debug);
  }

  return sequence;
}

std::vector<MotionPrimitiveListTree::Trajectory>
MotionPrimitiveListTree::getAllTrajectories(const FlatState& traj_start_state,
                                            const std::string& all_or_leaf) {
  std::vector<Sequence> sequences;
  if (all_or_leaf == "leaf_only" || all_or_leaf == "leaf")
    sequences = getAllLeafSequences();
  else
    sequences = getAllSequences();

  auto t_start = Clock::now();

  std::vector<Trajectory> trajectories;

  for (auto& sequence : sequences) {

    auto traj = forward_arc_primitive_trajectory::constructTrajectory(sequence, traj_start_state, 0.3, true);
    trajectories.push_back(traj);

  }

  std::chrono::duration<double> t_traj = Clock::now() - t_start;
  std::cout << "Time to reconstruct all trajectories: " << t_traj.count() << " s (" << t_traj.count() / sequences.size() << "s per traj)" << std::endl;
  return trajectories;

}

MotionPrimitiveListTree::Trajectory MotionPrimitiveListTree::pickAForwardTrajectory(const std::vector<MotionPrimitiveListTree::Trajectory>& trajectories, const FlatState& ref_state) {

  std::cout << "Picking a forward trajectory..." << std::endl;

  float min_cost = 10000;
  float cost = 0.0;
  Trajectory best_traj;

  // Find the lowest cost trajectory.

  for (auto &traj : trajectories) {
      // Reject the trajectory immediately if its too short
      float duration = 0;
      for (auto & seg : traj) duration += seg.duration();
      if (traj.size() <= 2 || duration < 2) continue;

      cost = cost::Deviation(traj);

      if (cost < min_cost) {
        min_cost = cost;
        best_traj = traj;
      }
  }
  return best_traj;
}

MotionPrimitiveListTree::Trajectory MotionPrimitiveListTree::pickJoystickTrajectory(const std::vector<MotionPrimitiveListTree::Trajectory>& trajectories, const Input& input)
{
  std::cout << "Picking a joystick trajectory..." << std::endl;
  // input needs to be in the form [vx, omega, vz, vside], assuming it is.

  float min_cost = 1e6;
  float cost = 0.0;
  Trajectory best_traj;

  // Find the lowest cost trajectory.

  for (auto &traj : trajectories) {

    // penalize the distance of the first trajectory.
    auto c0 = (traj.front().input() - input).norm()*3;

    // other costs
    auto c1 = cost::Input(traj, input, max_duration_) * 5;
    auto c2 = cost::Length(traj);
    cost = c0 + c1 + c2;

    if (cost < min_cost) {
      min_cost = cost;
      best_traj = traj;
    }
  }
  return best_traj;
}


MotionPrimitiveListTree::Trajectory MotionPrimitiveListTree::pickBestGlobalLocalTrajectory(const std::vector<MotionPrimitiveListTree::Trajectory>& trajectories,
  const std::vector<FlatState>& global,
  const std::vector<FlatState>& local)
{
  std::cout << "Picking the trajectory that is closest to the global local trajectories..." << std::endl;

  float min_cost = 1e6;
  float cost = 0.0;
  Trajectory best_traj;


  // weights on global and local closeness cost function
  double w_global = 1.0;
  double w_local = 0.0;

  // if either local or global trajectories contain 1 or no points, then turn off computing cost for it.
  if (global.size() < 2) w_global = 0.0;
  if (local.size() < 2) w_local = 0.0;

  std::cout << "w_global: " << w_global << " w_local: " << w_local << std::endl;

  // Get the duration for local and global trajectories for truncation
  double T_global = (global.size() < 2) ? 0.0 : global.back().t - global.front().t;
  double T_local = (local.size() < 2) ? 0.0 : local.back().t - local.front().t;

  // Get the arclength for local and global trajectories for truncation
  // double l_global = (global.size() < 2) ? 0.0 : path_utils::Length(global);
  // double l_local = (local.size() < 2) ? 0.0 : path_utils::Length(local);

  // Get the sample size for local and global trajectories, assuming they are evenly discretized
  double dt_global = (global.size() < 2) ? 0.0 : global[1].t - global[0].t;
  double dt_local = (local.size() < 2) ? 0.0 : local[1].t - local[0].t;

  // Find the lowest cost trajectory.

  for (auto &traj : trajectories) {

    double c_global = 0;
    double c_local = 0;
    if (global.size() >= 2)
    {
      // truncate the trajectory to have corresponding segments as the global trajectories.
      std::vector<FlatState> traj_sample = forward_arc_primitive_trajectory::samplePath(traj, dt_global);

      auto [duration, seg] = path_utils::getSegmentWithLength(traj_sample, traj_sample.front().pos, T_global);

      // compute the distance between each trajectory to the global trajectory.
      // Using Dynamic Time Warping Distnace
      // c_global = similarity::DynamicTimeWarpingDistance(seg, global);

      // Using Discrete Frechet Distance
      // limit the number of points to max 50 points.
      int num_points = std::min(50, (int)std::max(seg.size(), global.size()));
      auto traj1 = path_utils::discretizePath(seg, num_points);
      auto traj2 = path_utils::discretizePath(global, num_points);

      auto t_start = Clock::now();
      c_global = similarity::DiscreteFrechetDistance(traj1, traj2);
      std::chrono::duration<double> t_DFD = Clock::now() - t_start;
      // std::cout << "TIMING: DiscreteFrechetDistance: " << t_DFD.count() << " s for " << num_points << " points. Avg " << t_DFD.count()/num_points << " s per point" << std::endl;
    }

    if (local.size() >= 2)
    {
      // truncate the trajectory to have corresponding segments as the local trajectories.
      std::vector<FlatState> traj_sample = forward_arc_primitive_trajectory::samplePath(traj, dt_local);

      auto [duration, seg] = path_utils::getSegmentWithLength(traj_sample, traj_sample.front().pos, T_local);

      // compute the distance between each trajectory to the local trajectory.
      c_local = similarity::DynamicTimeWarpingDistance(seg, local);
    }

    double cost = w_local*c_local + w_global*c_global;

    if (cost < min_cost) {
      min_cost = cost;
      best_traj = traj;
    }
  }
  return best_traj;
}

MotionPrimitiveListTree::Trajectory MotionPrimitiveListTree::getMinCostTrajectory(const FlatState& start_state)
{

  // make a copy of tree_
  std::deque<Branch> trajectories_ = tree_;

  // Remove all of the trajectories that are 1 segment long
  vu::RemoveAllIf(trajectories_, [](Branch x){return std::get<2>(x) < 2;});

  // Find the min value of the bunch
  auto mincost_comparator = [](Branch left, Branch right)
    { return std::get<3>(left) < std::get<3>(right); };
  Branch min_cost_node = *std::min_element(trajectories_.begin(), trajectories_.end(), mincost_comparator);

  float min_cost = std::get<3>(min_cost_node);

  // Construct the sequence of lowest cost.
  Sequence sequence = getSequence(std::get<0>(min_cost_node));
  sequence.push_back(static_library_[std::get<1>(min_cost_node)]);

  // Construct trajectory
  auto traj = forward_arc_primitive_trajectory::constructTrajectory(sequence, start_state, 0.3, true);

  // Debug functions
  // push back min cost of the selected trajectory for analysis.
  min_cost_.push_back(min_cost);
  if (w_goal_ != 0) {
    auto dist = (traj.back().getFinalWorldPose().pos - goal_).norm();
    dist_to_goal_.push_back(dist);
  };

  // Write the min cost trajectory to file
  if (record_)
  {
    WriteAllTrajToFile(traj);
  }

  // Compute and save the jerk integral along the trajectory.
  float dt = 0.01;
  auto path = forward_arc_primitive_trajectory::samplePath(traj, dt);
  std::vector<float> jerk;
  jerk.reserve(path.size());
  for (auto & wpt : path)
  {
    jerk.push_back(wpt.jerk.norm());
  }
  auto jerk_integral = lu::Cumtrapz(dt, jerk);
  jerk_integral_.push_back(jerk_integral.back());


  return traj;

}
// ------------------------- PRINT FUNCTIONS --------------------------

void MotionPrimitiveListTree::printTree() {
  std::cout << "-------------Tree has " << tree_.size() << " nodes: " <<std::endl;
  std::cout << " (parent idx, primitive number in static tree, depth)" << std::endl;
  for (auto &node : tree_) {
    std::cout << " ( " << std::get<0>(node) << ", "
                       << std::get<1>(node) << ", "
                       << std::get<2>(node) << ")" << std::endl;
  }
  std::cout << "-------------end of tree." << std::endl;
}

void MotionPrimitiveListTree::printLeavesNotExpanded() {
  std::cout << "-------------sample_set_ has " << sample_set_.size() << " nodes: " <<std::endl;
  std::cout << " (parent idx, primitive number in static tree, depth)" << std::endl;
  for (auto &node : sample_set_) {
    std::cout << " ( " << std::get<0>(node) << ", "
                       << std::get<1>(node) << ", "
                       << std::get<2>(node) << ")" << std::endl;
  }
  std::cout << "-------------end of tree." << std::endl;
}

void MotionPrimitiveListTree::printTreeBranch(const std::vector<Branch>& branch_path) {
  // Branch(parent idx, primitive idx, depth):
  std::cout << "root ";
  for (auto & node: branch_path) {
    std::cout << "->("<<std::get<0>(node)<<", "<<std::get<1>(node)<<", " <<std::get<2>(node)<<")";
  }
  std::cout << std::endl;
}

void MotionPrimitiveListTree::printReverseTreeBranch(const std::vector<Branch>& branch_path) {
  for (auto & node: branch_path) {
    std::cout << "("<<std::get<0>(node)<<", "<<std::get<1>(node)<<", " <<std::get<2>(node)<<")->";
  }
  std::cout << " root "<< std::endl;
}

void MotionPrimitiveListTree::WriteAllTrajToFile(const std::vector<Trajectory>& trajectories)
{
  for (auto & traj : trajectories)
  {
    WriteAllTrajToFile(traj);
  }

  return;
}

void MotionPrimitiveListTree::WriteAllTrajToFile(const Trajectory& traj)
{
  // compute each vector and write it to file
  float dt = 0.01;
  auto path = forward_arc_primitive_trajectory::samplePath(traj, dt);

  std::vector<float> X, Y, Z, vel, acc, jerk;
  X.reserve(path.size());
  Y.reserve(path.size());
  Z.reserve(path.size());

  for (auto & wpt : path)
  {
    X.push_back(wpt.pos(0));
    Y.push_back(wpt.pos(1));
    Z.push_back(wpt.pos(2));
    vel_file_ << wpt.acc.norm() << ",";
    acc_file_ << wpt.acc.norm() << ",";
    jerk_file_ << wpt.jerk.norm() << ",";
  }
  vel_file_ << std::endl;
  acc_file_ << std::endl;
  jerk_file_ << std::endl;

  for (float x : X) {
    pos_file_ << x << ",";
  }
  pos_file_<< std::endl;

  for (float y : Y) {
    pos_file_ << y << ",";
  }
  pos_file_<< std::endl;

  for (float z : Z) {
    pos_file_ << z << ",";
  }
  pos_file_<< std::endl;

  return;
}

}; // namespace planner
