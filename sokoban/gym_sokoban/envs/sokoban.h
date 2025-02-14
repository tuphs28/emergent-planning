#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <algorithm>

using namespace std;

enum class roomStatus : unsigned char { wall, empty, box_not_on_tar, box_on_tar, player_not_on_tar, player_on_tar, tar, dan, box_on_dan, player_on_dan};
enum class action : unsigned char { noop, up, down, left, right };

void read_bmp(const string &img_dir, const string &img_name, vector<unsigned char> &data);
char roomStatus_to_char(const roomStatus r);

class Sokoban {	
public:
	Sokoban() = default;
	Sokoban(bool small, string level_dir, string img_dir, int level_num, int dan_num = 0, unsigned int seed = 0, bool mini = true, bool mini_unqtar = false, bool mini_unqbox = false) :
		player_pos_x(0),
		player_pos_y(0),
		box_left(0),
		tar_left(0),
		step_n(0),		
		img_x(small ? small_img_x : large_img_x),
		img_y(small ? small_img_x : large_img_x),
		obs_x(mini ? room_x: (small ? small_img_x : large_img_x)* room_x),
		obs_y(mini ? room_y: (small ? small_img_y : large_img_y)* room_y),
		obs_d(mini ? (mini_unqtar ? (mini_unqbox ? 13 : 10) : 7) : 3),
		obs_n(mini ? ((room_x-2) * (room_y-2) * (mini_unqtar ? (mini_unqbox ? 13 : 10) : 7)) :((small ? small_img_x * small_img_x : large_img_x * large_img_x)* room_x* room_y * 3)),
		level_dir(level_dir),
		img_dir(img_dir),
		done(false),
		small(small),
		mini(mini),
		mini_unqtar(mini_unqtar),
		mini_unqbox(mini_unqbox),
		room_status(),
		spirites(),
		tar_locs(),
		box_locs(),
		level_num(level_num),
		dan_num(dan_num),
		seed(seed){
		read_spirits();
	};
	static constexpr int room_x = 10, room_y = 10, small_img_x = 8, small_img_y = 8, large_img_x = 16, large_img_y = 16;
	void reset(unsigned char* obs);
	void reset_level(unsigned char* obs, const int room_id);
	void step(const action a, unsigned char* obs, float& reward, bool& done, bool& truncated_done, bool& cost);
	void step(const int a, unsigned char* obs, float& reward, bool& done, bool& truncated_done, bool& cost);
	int read_level(const int room_id);
	int print_level();
	void print_tar_locs();
	void print_box_locs();
	void clone_state(unsigned char* room_status, int& step_n, bool& done);
	void restore_state(const unsigned char* room_status, const int& step, const bool& done);
	int img_x, img_y, obs_x, obs_y, obs_n, step_n, obs_d;
	unsigned int seed;
	void set_seed(unsigned int seed);
private:
	float move(const action a);
	void move_player(roomStatus& old_r, roomStatus& new_r);
	void move_pos(const action a, int& x, int& y);
	float move_box(roomStatus& old_r, roomStatus& new_r);
	void read_spirits();
	void render(unsigned char* obs);
	int player_pos_x, player_pos_y, box_left, tar_left;
	bool done, small, mini, mini_unqtar, mini_unqbox;
	int level_num;
	int dan_num;
	string level_dir, img_dir;
	//default_random_engine defEngine;
	std::mt19937 defEngine;
	roomStatus room_status[room_y][room_x];
	vector<unsigned char> spirites[10];
	int tar_locs[4];
	int box_locs[4];
};