#include "sokoban.h"
#include <iostream>
using namespace std;


string level_dir = "/home/tom/mlmi/dissertation/thinker_private_planning/sokoban/gym_sokoban/envs/boxoban-levels/unfiltered/train";
string img_dir = "/home/tom/mlmi/dissertation/thinker_private_planning/sokoban/gym_sokoban/envs/surface";

int main()
{
	cout << "size of sokoban: " << sizeof(Sokoban) << endl;
	Sokoban sokoban(false, level_dir, img_dir, 1);
	unsigned char* obs = new unsigned char[sokoban.obs_n];
	float reward = 0.;
	bool done = false;
	sokoban.reset(obs);
	//sokoban.read_level(1002);;
	sokoban.print_level();
	int a;
	while (cin >> a) {
		sokoban.step(a, obs, reward, done, done, done);
		sokoban.print_level();
		cout << "reward: " << reward << " done: " << done << endl;
	}
	delete[] obs;
}
