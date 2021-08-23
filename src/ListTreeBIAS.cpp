
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

#include <atomic> // concurrency
#include <chrono>
#include <queue>
#include <stdexcept>

#include <codetimer_catkin/codetimer.h>

namespace cost = planner::cost_function;
namespace stats = stats_utils;
namespace su = sample_utils;
namespace vu = vector_utils;

using Clock = std::chrono::high_resolution_clock;
using NumLimitFloat = std::numeric_limits<float>;

namespace planner {


std::vector<int> MotionPrimitiveListTree::createAndSampleEliteSampleSet(const std::deque<float>& cost, int elite_set_size, int sample_batch_size)
{
  int K = std::min(elite_set_size, (int)cost.size());

  //Method 1: Sort and then add the first K values
  std::deque<float> cost_sorted;
  std::vector<size_t> idx_sorted;

  auto t_sort = Clock::now();
  vu::Sort(cost, &cost_sorted, &idx_sorted);
  CodeTimer::record("[ListTree::sampleFromEliteSampleSet] sort", t_sort);

  auto t_eliteset = Clock::now();
  std::vector<float> cost_eliteset;
  std::vector<int> idx_eliteset;
  cost_eliteset.reserve(K);
  idx_eliteset.reserve(K);
  for (unsigned int i = 0; i < K; i++)
  {
    idx_eliteset.push_back(idx_sorted[i]);
    float v = cost_sorted[i];
    if (softmax_enabled_) v = std::exp(beta_*v);
    cost_eliteset.push_back(v);
  }
  CodeTimer::record("[ListTree::sampleFromEliteSampleSet] elite set", t_eliteset);

  // Invert and sample from the elite set.

  std::vector<int> sampled_node_indices_from_vector = sampleFromEliteSampleSet(cost_eliteset, sample_batch_size);

  std::vector<int> sampled_node_indices;
  for (int node_idx : sampled_node_indices_from_vector)
  {
    sampled_node_indices.push_back(idx_eliteset[node_idx]);
  }

  return sampled_node_indices;
}

std::vector<MotionPrimitiveListTree::Branch> MotionPrimitiveListTree::createAndSampleEliteSampleSet(const std::deque<Branch>& sample_set, int elite_set_size, int sample_batch_size)
{
  int K = std::min(elite_set_size, (int)sample_set.size());

  auto t_eliteset = Clock::now();

  //Define a minpq parameter
  auto mincost_comparator_ = [](Branch left, Branch right)
    { return std::get<3>(left) > std::get<3>(right); };

  // Declare a priority queue.
  std::priority_queue<Branch, std::vector<Branch>,
    decltype(mincost_comparator_)> sample_set_minq(mincost_comparator_);

  // Add all elements in the sample set to the priority queue.
  for (auto & node : sample_set)
  {
    sample_set_minq.push(node);
  }
  CodeTimer::record("[ListTree::sampleFromEliteSampleSet] creating minpq", t_eliteset);

  // Get the K smallest value from it
  auto t_pop = Clock::now();
  std::vector<Branch> eliteset;
  std::vector<float> cost_eliteset;
  eliteset.reserve(K);
  cost_eliteset.reserve(K);
  for (int k = 0; k < K; k++)
  {
    auto node = sample_set_minq.top();
    eliteset.push_back(node);
    sample_set_minq.pop();
    cost_eliteset.push_back(std::get<3>(node));
  }
  CodeTimer::record("[ListTree::sampleFromEliteSampleSet] pop from minpq", t_pop);

  CodeTimer::record("[ListTree::sampleFromEliteSampleSet] elite set", t_eliteset);

  std::vector<int> sampled_node_indices_from_vector = sampleFromEliteSampleSet(cost_eliteset, sample_batch_size);

  std::vector<Branch> sampled_nodes;
  for (int node_idx : sampled_node_indices_from_vector)
  {
    sampled_nodes.push_back(eliteset[node_idx]);
  }

  return sampled_nodes;
}

std::vector<int> MotionPrimitiveListTree::sampleFromEliteSampleSet(const std::vector<float>& cost_eliteset, int sample_batch_size)
{
  // Invert and sample from the elite set.

  auto t_invert = Clock::now();
  std::vector<float> value;
  vu::Invert(cost_eliteset, &value);
  CodeTimer::record("[ListTree::sampleFromEliteSampleSet] invert", t_invert);

  auto t_norm = Clock::now();
  std::vector<float> normalized_value;
  vu::Normalize(value, &normalized_value);
  CodeTimer::record("[ListTree::sampleFromEliteSampleSet] normalize", t_norm);

  auto t_sample = Clock::now();
  std::vector<int> sampled_node_indices_from_vector =
    su::DiscreteSampleWithoutReplacement<float>(normalized_value,   sample_batch_size, generator_);

  CodeTimer::record("[ListTree::sampleFromEliteSampleSet] sample", t_sample);

  return sampled_node_indices_from_vector;
}


std::deque<int> MotionPrimitiveListTree::sampleFromEliteSampleSet(const std::deque<float>& cost_eliteset, int sample_batch_size)
{
  // Invert and sample from the elite set.

  auto t_invert = Clock::now();
  std::deque<float> value;
  vu::Invert(cost_eliteset, &value);
  CodeTimer::record("[ListTree::sampleFromEliteSampleSet] invert", t_invert);

  auto t_norm = Clock::now();
  std::deque<float> normalized_value;
  vu::Normalize(value, &normalized_value);
  CodeTimer::record("[ListTree::sampleFromEliteSampleSet] normalize", t_norm);

  auto t_sample = Clock::now();
  std::deque<int> sampled_node_indices_from_vector =
    su::DiscreteSampleWithoutReplacement<float>(normalized_value,   sample_batch_size, generator_);

  CodeTimer::record("[ListTree::sampleFromEliteSampleSet] sample", t_sample);

  return sampled_node_indices_from_vector;
}

// Monte Carlo Tree Search that expands all children, but only adds them to the leaf
bool MotionPrimitiveListTree::biasedTree(const FlatState& ref_state, const Eigen::Vector4d& input) {
  num_trees_created_++;
  test_input_ = input;

  //Define a minpq parameter
  std::function<bool(Branch, Branch)> mincost_comparator_ = [](Branch left, Branch right)
    { return std::get<3>(left) > std::get<3>(right); };

  // Declare a priority queue.
  std::priority_queue<Branch, std::vector<Branch>,
    decltype(mincost_comparator_)> sample_set_minq(mincost_comparator_);

  // Initialize loop variables
  float max_cost = NumLimitFloat::max();
  float cost_bound = NumLimitFloat::max();

  // Must initialize tree with this zero parameter.
  cost_.push_back(NumLimitFloat::max());
  sample_set_.emplace_back(0, 0, 0, 0.0);
  sample_set_minq.emplace(0, 0, 0, 0.0);

  std::cout << "Building biased tree" << std::flush;

  // If input cost is selected, compute a guiding primitive: Create a primitive with the same duration. input is of the form [vx, omega, vz, vside]
  Eigen::Vector3d input_vector;
  if (w_input_ != 0) {
    ForwardArcMotionPrimitives input_primitive(ref_state, input, max_duration_, 0.3, true);
    input_vector = (input_primitive.getFinalWorldPose().pos - input_primitive.getInitialWorldPose().pos).normalized();
  }

  std::atomic<int> num_processed = 0;
  std::atomic<int> added = 0;
  std::atomic<int> not_added = 0;
  int iter = 0;
  while (true) {
    auto t_start = Clock::now();
    iter++;

    // Check if we're stuck
    if ((cost_.empty() || sample_set_.empty()) && tree_.size() <= 1)
    {
      std::cout << "\n[ListTree] Tree is empty! no feasible nodes. Exiting" << std::endl;
      return false;
    }
    else if (cost_.empty() || sample_set_.empty())
    {
      // if no more nodes to sample, break there.
      iter_.push_back(iter);
      tree_initialized_ = true;
      std::cout << "[ListTree] No more nodes to sample, returning current tree" << std::endl;
      return true;
    }

    // ================= elite sample set ===============
    // Randomly select sample_batch_size number of nodes from the top `elite_set_size` number of nodes according to their cost.

    // std::vector<int> sampled_node_indices = sampleFromEliteSampleSet(sample_set_, elite_set_size_, sample_batch_size_);
    // std::vector<Branch> sampled_nodes = sampleFromEliteSampleSet(sample_set_, elite_set_size_, sample_batch_size_);

    int K = std::min(elite_set_size_, (int)sample_set_.size());

    auto t_eliteset = Clock::now();

    // Get the K smallest value from it
    auto t_pop = Clock::now();
    std::vector<Branch> eliteset;
    std::vector<float> cost_eliteset;
    eliteset.reserve(K);
    cost_eliteset.reserve(K);
    for (int k = 0; k < K; k++)
    {
      auto node = sample_set_minq.top();
      eliteset.push_back(node);
      sample_set_minq.pop();
      cost_eliteset.push_back(std::get<3>(node));
    }
    CodeTimer::record("[ListTree::sampleFromEliteSampleSet] pop from minpq", t_pop);

    std::vector<int> sampled_indices = sampleFromEliteSampleSet(cost_eliteset, sample_batch_size_);

    std::vector<Branch> sampled_nodes;
    for (int i = 0; i < eliteset.size(); i++)
    {
      bool found = (std::find(sampled_indices.begin(), sampled_indices.end(), i) != sampled_indices.end());

      // Add picked nodes to the set.
      if (found) sampled_nodes.push_back(eliteset[i]);
      // push nodes that were not picked back onto the queue
      else sample_set_minq.push(eliteset[i]);
    }

    CodeTimer::record("[ListTree::sampleFromEliteSampleSet] elite set", t_eliteset);

    // ========================== end elite sample set ===============

    // for (int sampled_node_idx : sampled_node_indices) {
    //
    //   // Add this node to the tree.
    //   auto leaf = sample_set_[sampled_node_idx];
    for (auto sampled_node : sampled_nodes) {

      // A little terminal progress indicator
      // if (tree_.size() % 10 == 0) std::cout << "." << std::flush;


      // Add this node to the tree.
      tree_.push_back(sampled_node);

      int parent_node_idx = tree_.size() - 1;

      // Get sequence associated with the parent node.
      FlatState parent_end_state;
      Sequence parent_sequence = getSequence(parent_node_idx);
      Trajectory parent_trajectory = forward_arc_primitive_trajectory::constructTrajectory(parent_sequence, ref_state, 0.3, true, &parent_end_state);

      // Get sequence associated with the just added node.
      Sequence sequence = parent_sequence;
      // We insert a dummy element which will be overwritten in the loop below.
      sequence.resize(sequence.size() + 1);
      Trajectory trajectory = parent_trajectory;
      trajectory.resize(trajectory.size() + 1);

      // For each static library, check for safety, and then evaluate safety.
      // Add the nodes to the Sample Set
      auto t_children = Clock::now();
      std::atomic<int> num_unsafe = 0;

      /******************     BEGIN MULTITHREADING        *********************/

      std::mutex write_lock;

      // A truly hacky way to do this parallel children evaluation (instead of making nice functions).
      auto thread_f = [this, sequence, trajectory, &sample_set_minq, parent_node_idx, &num_unsafe, &added, &not_added, parent_end_state, &num_processed, input, ref_state, cost_bound, &write_lock, &input_vector](int offset) mutable {
      for (size_t i = offset; i < static_library_.size(); i += num_threads_) {

        auto t_eachprimitive = Clock::now();

        // Make new node
        Branch new_node(parent_node_idx, i, std::get<2>(tree_[parent_node_idx])+1, 0.0);

        // ref_state hsould be the end state of the previous trajectory...
        // auto primitive = forward_arc_primitive_trajectory::generatePrimitive(static_library_[i], parent_end_state, 0.3, true);
        auto branch_input = std::get<0>(static_library_[i]);
        auto branch_duration = std::get<1>(static_library_[i]);

        // augment the branch input velocity term with the current velocity of
        // the joystick, to enable adaptive tree size
        branch_input(0) = input(0);

        auto primitive = ForwardArcMotionPrimitives(parent_end_state, branch_input, branch_duration, 0.3, true);

        // Check for safety (prune)

        bool safe = true;
        auto t_safety = Clock::now();
        safe = collision_checker_->checkTrajectorySafe(&primitive);
        CodeTimer::record("[ListTree] checkTrajectorySafe", t_safety);

        num_processed++;

        // if it's not safe or already in tree, continue the expansion.
        if (!safe) {
          num_unsafe++;
          CodeTimer::record("[ListTree] process one child - unsafe", t_eachprimitive);
          continue;
        }

        /* 3. SIMULATION: Evaluate the cost for the  (according to some cost function) */

        // Compute cost by evaluating the trajectory as a whole.

        // Get sequence associated with the just added node.
        sequence[sequence.size() - 1] = static_library_[i];
        trajectory[sequence.size() - 1] = primitive;

        // compute Cost
        auto t_cost = Clock::now();
        float cost = 0;

        float c_sl = 0, c_duration = 0, c_speed = 0, c_smooth = 0, c_goal = 0, c_point = 0, c_direction = 0, c_input = 0, c_length = 0, c_deviation = 0;

        if (w_sl_ != 0) {
          auto t_sl = Clock::now();
          c_sl =  cost::StraightLine(trajectory);
          CodeTimer::record("[ListTree::cost] StraightLine", t_sl);
        }

        if (w_smooth_ != 0) {
          auto t_smooth = Clock::now();
          c_smooth = cost::Smoothness(trajectory);
          CodeTimer::record("[ListTree::cost] Smoothness", t_smooth);
        }

        if (w_direction_ != 0) {
          auto t_dev = Clock::now();
          c_direction = cost::Direction(trajectory, ref_state);
          CodeTimer::record("[ListTree::cost] Direction", t_dev);
        }

        if (w_input_ != 0) {
          auto t_dir = Clock::now();
          // c_input = cost::Input(trajectory, input);
          // c_input = cost::Input(trajectory, input, max_duration_);
          c_input = cost::Input2(trajectory, input_vector);
          CodeTimer::record("[ListTree::cost] Input", t_dir);
        }

        // trajectory length characteristics
        if (w_duration_ != 0) {
          auto t_dur = Clock::now();
          c_duration = cost::Duration(trajectory);
          CodeTimer::record("[ListTree::cost] Duration", t_dur);
        }

        if (w_speed_ != 0) {
          auto t_speed = Clock::now();
          c_speed = cost::Speed(trajectory);
          CodeTimer::record("[ListTree::cost] Speed", t_speed);
        }

        if (w_length_ != 0) {
          auto t_length = Clock::now();
          c_length = cost::Length(trajectory);
          CodeTimer::record("[ListTree::cost] Length", t_length);
        }

        // state-space costs

        if (w_goal_ != 0) {
          auto t_goal = Clock::now();
          c_goal = cost::Goal(trajectory, goal_);
          CodeTimer::record("[ListTree::cost] goal", t_goal);
        }

        if (w_point_ != 0) {
          auto t_point = Clock::now();
          c_point = cost::Point(trajectory, goal_, 48);
          CodeTimer::record("[ListTree::cost] point", t_point);
        }

        if (w_deviation_ != 0) {
          auto t_deviation = Clock::now();
          c_deviation = cost::Deviation(trajectory);
          CodeTimer::record("[ListTree::cost] deviation", t_deviation);
        }

        cost = w_speed_*c_speed + w_duration_*c_duration
             + w_length_ * c_length +
             + w_sl_*c_sl + w_smooth_*c_smooth
             + w_direction_*c_direction + w_input_*c_input
             + w_goal_*c_goal + w_point_*c_point + w_deviation_ * c_deviation;

        CodeTimer::record("[ListTree::cost] total", t_cost);

        if (cost_bound_enabled_ && cost > cost_bound) {
          not_added++;
          CodeTimer::record("[ListTree] process one child - cost computed, not added", t_eachprimitive);
          continue;
        }

        // Add it to the tree
        std::get<3>(new_node) = cost;

        // Add new child to sample set
        {
          std::lock_guard<std::mutex> lock(write_lock);
          sample_set_.push_back(new_node);
          sample_set_minq.push(new_node);
          cost_.push_back(cost);
        }

        added++;

        // Re-adjust the max cost

        CodeTimer::record("[ListTree] process one child - added", t_eachprimitive);

      } // End all new samples.

      };

      std::vector<std::thread> threads;
      for (int i = 0; i < num_threads_; i++) {
        threads.push_back(std::thread(std::bind(thread_f, i)));
      }

      for (auto &thread : threads) {
        thread.join();
      }

      /******************     END MULTITHREADING        *********************/

      CodeTimer::record("[ListTree] process all children", t_children);

      // If the number of unsafe elements are the same as the child size, don't add this element.
      if (num_unsafe == static_library_.size())
      {
        tree_.pop_back();
      }

      // Remove all elements from the list.
      vu::RemoveAll(sample_set_, sampled_node);

    } // end sampled_node_indices

    // Remove this list of expanded nodes from the list of leaves (and same with the cost)
    // auto t_removeindices = Clock::now();
    // vu::RemoveAtIndices(sample_set_, sampled_node_indices);
    // CodeTimer::record("[ListTree] vu::RemoveAtIndices", t_removeindices);
    // vu::RemoveAtIndices(cost_, sampled_node_indices);

    // Truncate

    if (cost_bound_enabled_) {
      // Method 1: by computing percentile. This is quite expensive.
      // auto t_percentile = Clock::now();
      // max_cost = stats::PercentileValue(cost_, percentile_);
      // CodeTimer::record("[ListTree] stats::PercentileValue", t_percentile);
      // cost_bound = max_cost;

      // Method 2: Set max cost. Cheaper
      auto t_max = Clock::now();
      // max_cost = vu::Max(cost_); // Method 1
      ///// Method 2
      auto maxcost_comparator_ = [](Branch left, Branch right)
        { return std::get<3>(left) < std::get<3>(right); };
      Branch max_cost_node = *std::max_element(sample_set_.begin(), sample_set_.end(), maxcost_comparator_);
      float max_cost = std::get<3>(max_cost_node);
      cost_bound = percentile_ * max_cost;
      CodeTimer::record("[ListTree] vu::Max", t_max);
    }

    /* 5. TERMINATING CONDITION: what should this be? TODO */
    if (tree_.size() >= tree_size_) {
      std::cout << std::endl << "Tree have reached size " << tree_.size() << ", exiting mcts" << std::endl;
      break;
    }

    CodeTimer::record("[ListTree] per iter", t_start);

  } // end one iteration

  num_nodes_processed_.push_back(num_processed);
  nodes_added_.push_back(added);
  nodes_rejected_.push_back(not_added);
  iter_.push_back(iter);
  tree_initialized_ = true;

  std::cout << std::endl;

  return true;
}


}; // namespace planner
