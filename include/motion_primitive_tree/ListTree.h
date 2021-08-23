/*
MotionPrimitiveListTree
Copyright (C) 2019 Xuning Yang

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <deque>
#include <vector>

#include <motion_primitive_tree/CostFunctionWeightsConfig.h>

#include <dynamic_reconfigure/server.h>

#include <cpp_utils/print_utils.h>
#include <cpp_utils/sample_utils.h>
#include <cpp_utils/stats_utils.h>
#include <cpp_utils/vector_utils.h>
#include <forward_arc_primitives/ForwardArcMotionPrimitives.h>
#include <forward_arc_primitives/ForwardArcPrimitiveTrajectory.h>
#include <collision_checker/CollisionChecker.h>
#include <trajectory_utils/PathUtils.h>
#include <trajectory_utils/Similarity.h>

namespace planner {

/* Class MotionPrimitiveListTree is a motion primitive tree that stores the
tree in three objects:

1.  static_library (ST): A motion primitive library of one level, created by
    combinations of control inputs. ST is a list of PrimitiveInputs, where:

    PrimitiveInputs = Eigen::Vector4d(linear vel, angular vel, z vel, duration)

2. tree (T): A list of tuples that contains the tree structure.
   Each tuple is represented as a single Branch, like so:

    Branch = (parent index in T, index of primitive in the ST, current depth, cost)

3. leaves (L): a list of all the leaves in the tree, represented by their
   index in T.
*/

class MotionPrimitiveListTree
{
public:
  MotionPrimitiveListTree();
  ~MotionPrimitiveListTree();

  using Index = unsigned int;
  using Branch = std::tuple<Index, Index, Index, float>;
  using Input = forward_arc_primitive_trajectory::Input; // Eigen::Vector4d
  using TrajectoryParam = forward_arc_primitive_trajectory::TrajectoryParam; // tuple<input, duration>
  using Sequence = std::deque<TrajectoryParam>; // segments of primitives
  using Trajectory = forward_arc_primitive_trajectory::Trajectory;

  std::vector<TrajectoryParam> static_library_;
  std::deque<Index> leaves_;
  std::deque<Branch> tree_;
  std::deque<float> cost_;
  std::deque<Branch> sample_set_;

  void initialize(const ros::NodeHandle& n,
      const std::vector<double>& x_vel,
      const std::vector<double>& omega_vel,
      const std::vector<double>& z_vel,
      const std::vector<double>& duration);

  // if an existing version of collision_checker is provided
  void initialize(const ros::NodeHandle& n,
      const std::vector<double>& x_vel,
      const std::vector<double>& omega_vel,
      const std::vector<double>& z_vel,
      const std::vector<double>& duration,
      const std::shared_ptr<CollisionChecker>& collision_checker);

  void setUpFileWriting(const std::string& saveto_directory);

  void clear();

  // Tree building Methods

  bool buildTreeToDepth(unsigned int depth);
  bool mcts(const FlatState& ref_state, const Eigen::Vector4d& input = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0));
  void mcts2(const FlatState& ref_state, int max_nodes);           // MCTS with batch expansion
  bool biasedTree(const FlatState& ref_state, const Eigen::Vector4d& input = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0)); // biased tree generation

  bool buildTreeFromNodes(const FlatState& ref_state, int max_nodes, float percentile, const std::deque<Branch>& initial_nodes, const std::deque<float>& initial_costs);

  std::vector<Branch> createAndSampleEliteSampleSet(const std::deque<Branch>& sample_set, int K, int sample_batch_size);
  std::vector<int> createAndSampleEliteSampleSet(const std::deque<float>& cost, int K, int sample_batch_size);
  std::vector<int> sampleFromEliteSampleSet(const std::vector<float>& cost_eliteset, int sample_batch_size);
  std::deque<int> sampleFromEliteSampleSet(const std::deque<float>& cost_eliteset, int sample_batch_size);

  // Methods for generating Sequential motion primitives

  Sequence getSequence(unsigned int leaf);
  std::vector<Sequence> getAllSequences();
  std::vector<Sequence> getAllLeafSequences();

  // Methods for selecting trajectories

  std::vector<Trajectory> getAllTrajectories(const FlatState& traj_start_state, const std::string &all_or_leaf = "all");

  MotionPrimitiveListTree::Trajectory pickAForwardTrajectory(const std::vector<Trajectory>& trajectories, const FlatState& ref_state);

  MotionPrimitiveListTree::Trajectory pickJoystickTrajectory(const std::vector<Trajectory>& trajectories, const Input& input);

  MotionPrimitiveListTree::Trajectory pickBestGlobalLocalTrajectory(const std::vector<MotionPrimitiveListTree::Trajectory>& trajectories,
    const std::vector<FlatState>& global,
    const std::vector<FlatState>& local);

  MotionPrimitiveListTree::Trajectory getMinCostTrajectory(const FlatState& start_state);

  void printTree();
  void printLeavesNotExpanded();
  void printTreeBranch(const std::vector<Branch>& branch_path);
  void printReverseTreeBranch(const std::vector<Branch>& branch_path);

private:

  int num_threads_;
  bool tree_initialized_;
  int tree_size_;

  int elite_set_size_;
  int sample_batch_size_;

  bool softmax_enabled_;
  float beta_;

  bool cost_bound_enabled_;
  float percentile_;

  double max_duration_;
  double max_xvel_;
  double max_zvel_;
  double max_omega_;

  Eigen::Vector3d goal_;
  Eigen::Vector4d test_input_;

  float w_smooth_, w_sl_, w_duration_, w_goal_, w_direction_, w_point_, w_speed_, w_input_, w_length_, w_deviation_;


  std::mt19937 generator_;

  std::shared_ptr<CollisionChecker> collision_checker_;

  std::unique_ptr<dynamic_reconfigure::Server<motion_primitive_tree::CostFunctionWeightsConfig>> cost_function_dynamic_reconfigs_server_;

  // Debug/Timing info
  int num_trees_created_ = 0;
  std::vector<int> num_nodes_processed_;
  std::vector<int> nodes_added_;
  std::vector<int> nodes_rejected_;
  std::vector<int> iter_;
  std::vector<int> depth_of_tree_;
  std::vector<float> min_cost_;
  std::vector<float> dist_to_goal_;
  std::vector<float> jerk_integral_;

  std::vector<int> all_trajectories_generated_;
  std::vector<int> leaf_trajectories_generated_;

  bool record_ = false;

  void costFunctionWeightReconfigureCallback(motion_primitive_tree::CostFunctionWeightsConfig &cfg, uint32_t levels);
};

} // namespace planner
