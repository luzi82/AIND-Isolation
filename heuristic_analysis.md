# Heuristic Analysis

## tournament.py modification

The file `tournament2.py` is copy of `tournament.py`, and is changed in following ways:

- Increase NUM_MATCHES from 5 to 25, so more matches will be done.
- Remove `ID_Improved` from test_agents, since `ID_Improved` is only needed to run once to get it's performance value.

## Heuristic functions

### ID_Improved (given)

`ID_Improved` calculate the number of available move of 2 players, and minus them to get the score.

The `tournament2.py` result is as follow:

    *************************
     Evaluating: ID_Improved 
    *************************
    
    Playing Matches:
    ----------
      Match 1: ID_Improved vs   Random      Result: 93 to 7
      Match 2: ID_Improved vs   MM_Null     Result: 77 to 23
      Match 3: ID_Improved vs   MM_Open     Result: 54 to 46
      Match 4: ID_Improved vs MM_Improved   Result: 44 to 56
      Match 5: ID_Improved vs   AB_Null     Result: 57 to 43
      Match 6: ID_Improved vs   AB_Open     Result: 34 to 66
      Match 7: ID_Improved vs AB_Improved   Result: 50 to 50
    
    
    Results:
    ----------
    ID_Improved         58.43%

<table border=1>
<tr><th></th><th>Random</th><th>MM_Null</th><th>MM_Open</th><th>MM_Improved</th><th>AB_Null</th><th>AB_Open</th><th>AB_Improved</th><th>Result</th></tr>
<tr><th>ID_Improved</th><td>93</td><td>77</td><td>54</td><td>44</td><td>57</td><td>34</td><td>50</td><td>58.43%</td></tr>
</table>

### custom_score_0 (Improved ID_Improved)

The `custom_score_0` put n moves forward into consideration.

For example, in a 4x4 game:

![](img/sample_game.png)

We first do breadth first search to find number of move required to go to each cell, in empty board.

Since the calculation is based on empty board, the result can be cached to save CPU.

![](img/cs0_move_a.png)
![](img/cs0_move_b.png)

Then, for each cell which require n moves, we score it r<sub>0</sub><sup>n</sup>.

For r<sub>0</sub> = 1/8:

![](img/cs0_score_a.png)
![](img/cs0_score_b.png)

Finally, we sum up all cell which is not blocked, and subtract each other.

- Player A: value = 0.316
- Player B: value = 0.193
- score = 0.316 - 0.193 = 0.123

The `tournament2.py` result is as follow:

<table border=1>
<tr><th>r<sub>0</sub></th><th>Random</th><th>MM_Null</th><th>MM_Open</th><th>MM_Improved</th><th>AB_Null</th><th>AB_Open</th><th>AB_Improved</th><th>Result</th></tr>
<tr><th>1/2</th><td>95</td><td>86</td><td>57</td><td>47</td><td>75</td><td>50</td><td>56</td><td>66.57%</td></tr>
<tr><th>1/3</th><td>94</td><td>80</td><td>44</td><td>50</td><td>70</td><td>60</td><td>42</td><td>62.86%</td></tr>
<tr><th>1/4</th><td>93</td><td>82</td><td>51</td><td>53</td><td>65</td><td>51</td><td>46</td><td>63.00%</td></tr>
<tr><th>1/5</th><td>95</td><td>88</td><td>54</td><td>41</td><td>72</td><td>58</td><td>61</td><td>67.00%</td></tr>
<tr style='color:red'><th>1/6</th><td>86</td><td>83</td><td>62</td><td>54</td><td>72</td><td>58</td><td>58</td><td>67.57%</td></tr>
<tr><th>1/7</th><td>91</td><td>83</td><td>48</td><td>53</td><td>73</td><td>54</td><td>53</td><td>65.00%</td></tr>
<tr><th>1/8</th><td>98</td><td>79</td><td>51</td><td>47</td><td>77</td><td>55</td><td>51</td><td>65.43%</td></tr>
<tr><th>1/9</th><td>94</td><td>82</td><td>54</td><td>52</td><td>72</td><td>53</td><td>56</td><td>66.14%</td></tr>
<tr><th>1/10</th><td>92</td><td>77</td><td>54</td><td>57</td><td>68</td><td>56</td><td>57</td><td>65.86%</td></tr>
</table>

It is sightly better than `ID_Improved`.
The differences between each r values are too small to make any conclusion.

### custom_score_1 (Neural network)

The `custom_score_1` is based on neural network and minimax Q learning.

First, we convert the 7x7 game data into 7x7x3 = 147 boolean value.
The first 7x7 boolean value represent which cell is not blocked.
The second 7x7 boolean value represent the location of active player.
The third 7x7 boolean value represent the location of inactive player.

![](img/sample_game.png)

becomes

![](img/cs1_data.png)

Then we put the values into 3 hidden layers neural network, which represent the score of 8 move.
We apply boolean mask to filter out impossible move.

The score of state s would be:

    max( Q(s0,a0) for all a0 )

The neural network is trained by following equation:

    Q(s0,a0) = -1                                   if loss
             = +1                                   if win
             = - gamma * max( Q(s1,a1) for all a1 ) otherwise

Since the Q function return the score of the active player, the right hand side of the equation should be negative, since the second move is made by opponent.

Step of training:

0. Make 100000 moves
0. Train upon 1-100000th move
0. Make 100001st move
0. Train upon 2-100001th move
0. Make 100002st move
0. Train upon 3-100002th move
0. continue...

The `tournament2.py` result is as follow:

<table border=1>
<tr><th>moves</th><th>Random</th><th>MM_Null</th><th>MM_Open</th><th>MM_Improved</th><th>AB_Null</th><th>AB_Open</th><th>AB_Improved</th><th>Result</th></tr>
<tr><th>200000</th><td>86</td><td>57</td><td>36</td><td>37</td><td>58</td><td>31</td><td>36</td><td>48.71%</td></tr>
<tr><th>300000</th><td>87</td><td>72</td><td>44</td><td>36</td><td>59</td><td>37</td><td>44</td><td>54.14%</td></tr>
<tr><th>400000</th><td>86</td><td>70</td><td>32</td><td>38</td><td>54</td><td>37</td><td>36</td><td>50.43%</td></tr>
<tr><th>500000</th><td>85</td><td>69</td><td>35</td><td>34</td><td>62</td><td>40</td><td>44</td><td>52.71%</td></tr>
<tr><th>600000</th><td>94</td><td>67</td><td>34</td><td>28</td><td>56</td><td>37</td><td>49</td><td>52.14%</td></tr>
<tr><th>700000</th><td>89</td><td>71</td><td>40</td><td>29</td><td>53</td><td>32</td><td>43</td><td>51.00%</td></tr>
</table>

The result is disappointing.  Here are possible reasons:

- Not enough sample
- The neural network is too simple

Possible improvement:

- Increase the number of sample window size and sample number.  Require more training time.
- Increase the complexity of neural network.  Apply convolution layer, add more feature to input data set.  Require more CPU / GPU time.

Moreover, the trained neural network can be used only in game same as training game.
For games which have difference size, it is necessary to train another neural network.

The `custom_score_1` require TensorFlow to run.


### custom_score_2 (Distance from center)

The output is the distance between player location to the center.
Since we are not sure it is better to stay at center or edge, so we make 2 opposite functions.

- version 2a: Self distance - Opponent distance.  Stay on edge will get higher score.
- version 2b: Opponent distance - Self distance.  Stay in center will get higher score.

<table border=1>
<tr><th></th><th>Random</th><th>MM_Null</th><th>MM_Open</th><th>MM_Improved</th><th>AB_Null</th><th>AB_Open</th><th>AB_Improved</th><th>Result</th></tr>
<tr><th>2a</th><td>86</td><td>38</td><td>16</td><td>18</td><td>25</td><td>22</td><td>17</td><td>31.71%</td></tr>
<tr style='color:red'><th>2b</th><td>82</td><td>71</td><td>37</td><td>37</td><td>54</td><td>45</td><td>42</td><td>52.57%</td></tr>
</table>

Even though the result is disappointing, the function is too simple and it achieve 52.57%.
Moreover the difference of 2a and 2b is high, we may assume that it is better to be in center.
The discovery can help to develop other score function.

### custom_score_3 ( custom_score_0 + custom_score_2 )

Since we know `custom_score_0` and `custom_score_2b` are good heuristic functions, so we combine those functions to make a new function.

    custom_score_3b = custom_score_0(r=1/6) + r3 * custom_score_2b

<table border=1>
<tr><th>r<sub>3b</sub></th><th>Random</th><th>MM_Null</th><th>MM_Open</th><th>MM_Improved</th><th>AB_Null</th><th>AB_Open</th><th>AB_Improved</th><th>Result</th></tr>
<tr><th>0</th><td>86</td><td>83</td><td>62</td><td>54</td><td>72</td><td>58</td><td>58</td><td>67.57%</td></tr>
<tr><th>0.02</th><td>93</td><td>79</td><td>48</td><td>53</td><td>72</td><td>55</td><td>55</td><td>65.00%</td></tr>
<tr><th>0.04</th><td>92</td><td>81</td><td>50</td><td>51</td><td>65</td><td>56</td><td>51</td><td>63.71%</td></tr>
<tr><th>0.06</th><td>92</td><td>77</td><td>50</td><td>57</td><td>66</td><td>55</td><td>46</td><td>63.29%</td></tr>
<tr><th>0.08</th><td>90</td><td>75</td><td>56</td><td>56</td><td>64</td><td>50</td><td>35</td><td>60.86%</td></tr>
<tr><th>0.10</th><td>94</td><td>74</td><td>56</td><td>54</td><td>74</td><td>48</td><td>54</td><td>64.86%</td></tr>
<tr><th>0.12</th><td>95</td><td>76</td><td>56</td><td>58</td><td>69</td><td>51</td><td>45</td><td>64.29%</td></tr>
<tr><th>0.14</th><td>90</td><td>73</td><td>58</td><td>49</td><td>68</td><td>55</td><td>44</td><td>62.43%</td></tr>
<tr><th>0.16</th><td>92</td><td>76</td><td>47</td><td>37</td><td>67</td><td>51</td><td>47</td><td>59.57%</td></tr>
<tr><th>0.18</th><td>96</td><td>73</td><td>50</td><td>51</td><td>66</td><td>55</td><td>51</td><td>63.14%</td></tr>
<tr><th>0.20</th><td>96</td><td>83</td><td>50</td><td>45</td><td>64</td><td>49</td><td>45</td><td>61.71%</td></tr>
</table>

It seems `custom_score_2b` bring negative effect to `custom_score_0`.

Here is the test to combine `custom_score_0` and `custom_score_2a`

    custom_score_3a = custom_score_0(r=1/6) + r3 * custom_score_2a

<table border=1>
<tr><th>r<sub>3a</sub></th><th>Random</th><th>MM_Null</th><th>MM_Open</th><th>MM_Improved</th><th>AB_Null</th><th>AB_Open</th><th>AB_Improved</th><th>Result</th></tr>
<tr><th>0</th><td>86</td><td>83</td><td>62</td><td>54</td><td>72</td><td>58</td><td>58</td><td>67.57%</td></tr>
<tr><th>0.02</th><td>90</td><td>77</td><td>53</td><td>54</td><td>61</td><td>52</td><td>44</td><td>61.57%</td></tr>
<tr><th>0.04</th><td>93</td><td>77</td><td>56</td><td>54</td><td>76</td><td>53</td><td>48</td><td>65.29%</td></tr>
<tr><th>0.06</th><td>93</td><td>80</td><td>57</td><td>56</td><td>67</td><td>54</td><td>53</td><td>65.71%</td></tr>
<tr><th>0.08</th><td>98</td><td>81</td><td>60</td><td>51</td><td>73</td><td>57</td><td>51</td><td>67.29%</td></tr>
<tr><th>0.10</th><td>90</td><td>76</td><td>56</td><td>50</td><td>67</td><td>49</td><td>45</td><td>61.86%</td></tr>
<tr><th>0.12</th><td>95</td><td>83</td><td>46</td><td>47</td><td>74</td><td>53</td><td>46</td><td>63.43%</td></tr>
<tr><th>0.14</th><td>91</td><td>91</td><td>55</td><td>46</td><td>60</td><td>47</td><td>44</td><td>62.00%</td></tr>
<tr><th>0.16</th><td>93</td><td>86</td><td>50</td><td>48</td><td>71</td><td>52</td><td>45</td><td>63.57%</td></tr>
<tr><th>0.18</th><td>95</td><td>75</td><td>50</td><td>43</td><td>76</td><td>45</td><td>41</td><td>60.71%</td></tr>
<tr><th>0.20</th><td>97</td><td>80</td><td>51</td><td>40</td><td>78</td><td>49</td><td>45</td><td>62.86%</td></tr>
</table>

No improvement found also.

### custom_score_4 (Simulation)

`custom_score_4` run simulation play to end game, and output the win/loss of the result.

In simulation, the game will choose the move closest to the board center.  We choose this feature since it is less CPU consumption and effective.

For the game which win in n step, the function would output r<sub>4</sub><sup>n</sup> (0 &lt; r<sub>4</sub> &lt; 1).
For the game which loss in n step, the function would output -r<sub>4</sub><sup>n</sup>.
So the AI would prefer early win than later win, and prefer later loss than early loss.

The runtime and performance of this function depends on number of blank cell in the board.
With more blank cell, simulation require more time, and the result is less accurate.
As step count increase, +/-r<sub>4</sub><sup>n</sup> would be close to zero.
So we may put a cutdown to save CPU.  But now we just simulate to endgame.

It is meaningless to tune r<sub>4</sub>.  We just need to ensure 0 &lt; r<sub>4</sub> &lt; 1.

<table border=1>
<tr><th>type</th><th>Random</th><th>MM_Null</th><th>MM_Open</th><th>MM_Improved</th><th>AB_Null</th><th>AB_Open</th><th>AB_Improved</th><th>Result</th></tr>
<tr><th>cs4</th><td>95</td><td>81</td><td>47</td><td>51</td><td>71</td><td>53</td><td>52</td><td>64.29%</td></tr>
</table>

It's strength is close to `ID_Improved`, and weaker than `custom_score_0`, even though it consume high CPU.

### custom_score_5 (custom_score_0 + custom_score_4)

`custom_score_5` combine `custom_score_0` and `custom_score_4` by adding them together.

    custom_score_5 = (1-r5) * custom_score_0 + r5 * custom_score_4

To get the best performance of the function, we need to tune r<sub>0</sub>, r<sub>4</sub> and r<sub>5</sub> together.
For simplicity we take r<sub>0</sub> = 1/6, r<sub>4</sub> = 0.99, and tune r<sub>5</sub>.

Result: (r<sub>0</sub> = 1/6, r<sub>4</sub> = 0.99, NUM_MATCHES = 250)

<table border=1>
<tr><th>r<sub>5</sub></th><th>Random</th><th>MM_Null</th><th>MM_Open</th><th>MM_Improved</th><th>AB_Null</th><th>AB_Open</th><th>AB_Improved</th><th>Result</th></tr>
<tr><th>0.1</th><td>968</td><td>868</td><td>652</td><td>627</td><td>765</td><td>619</td><td>603</td><td>72.89%</td></tr>
<tr><th>0.2</th><td>958</td><td>879</td><td>672</td><td>613</td><td>790</td><td>645</td><td>606</td><td>73.76%</td></tr>
<tr style='color:red'><th>0.3</th><td>974</td><td>905</td><td>671</td><td>669</td><td>808</td><td>625</td><td>617</td><td>75.27%</td></tr>
<tr><th>0.4</th><td>964</td><td>886</td><td>674</td><td>627</td><td>787</td><td>630</td><td>587</td><td>73.64%</td></tr>
<tr><th>0.5</th><td>965</td><td>877</td><td>668</td><td>602</td><td>794</td><td>611</td><td>619</td><td>73.37%</td></tr>
<tr><th>0.6</th><td>958</td><td>890</td><td>686</td><td>630</td><td>782</td><td>637</td><td>609</td><td>74.17%</td></tr>
<tr><th>0.7</th><td>965</td><td>867</td><td>681</td><td>620</td><td>820</td><td>617</td><td>599</td><td>73.84%</td></tr>
<tr><th>0.8</th><td>958</td><td>887</td><td>647</td><td>590</td><td>801</td><td>642</td><td>592</td><td>73.10%</td></tr>
<tr><th>0.9</th><td>968</td><td>894</td><td>644</td><td>614</td><td>795</td><td>633</td><td>589</td><td>73.39%</td></tr>
</table>

The performance of `custom_score_5` is far better than `custom_score_0` and `custom_score_4`.
r<sub>0</sub>, r<sub>4</sub>, and r<sub>5</sub> can be further tuned to achieve better performance.

## Conclusion

Here is the summary of the heuristic functions:

<table border=1>
<tr><th>type</th><th>Random</th><th>MM_Null</th><th>MM_Open</th><th>MM_Imp</th><th>AB_Null</th><th>AB_Open</th><th>AB_Imp</th><th>Result</th></tr>
<tr><th>ID_Improved</th><td>93</td><td>77</td><td>54</td><td>44</td><td>57</td><td>34</td><td>50</td><td>58.43%</td></tr>
<tr><th>custom_0</th><td>86</td><td>83</td><td>62</td><td>54</td><td>72</td><td>58</td><td>58</td><td>67.57%</td></tr>
<tr><th>custom_1</th><td>89</td><td>71</td><td>40</td><td>29</td><td>53</td><td>32</td><td>43</td><td>51.00%</td></tr>
<tr><th>custom_2b</th><td>82</td><td>71</td><td>37</td><td>37</td><td>54</td><td>45</td><td>42</td><td>52.57%</td></tr>
<tr><th>custom_3b</th><td>93</td><td>79</td><td>48</td><td>53</td><td>72</td><td>55</td><td>55</td><td>65.00%</td></tr>
<tr><th>custom_4</th><td>95</td><td>81</td><td>47</td><td>51</td><td>71</td><td>53</td><td>52</td><td>64.29%</td></tr>
<tr style='color:red'><th>custom_5</th><td>97.4</td><td>90.5</td><td>67.1</td><td>66.9</td><td>80.8</td><td>62.5</td><td>61.7</td><td>75.27%</td></tr>
</table>

We recommend `custom_score_5`:

- It achieve best result when matching with random opponent, with chance 97.4%.
- It achieve best result when matching with intelligent opponents such as `open_move_score` and `improved_score`, with higher than 60% chance.
- It's performance is the best among all other custom score.

In CPU limited environment, we recommend `custom_score_0`, as it can run in constant time in larger board.

In extreme limited environment, `custom_score_2b` would be a choice as it does not require much calculation.

tournament.py output:

    This script evaluates the performance of the custom heuristic function by
    comparing the strength of an agent using iterative deepening (ID) search with
    alpha-beta pruning against the strength rating of agents using other heuristic
    functions.  The `ID_Improved` agent provides a baseline by measuring the
    performance of a basic agent using Iterative Deepening and the "improved"
    heuristic (from lecture) on your hardware.  The `Student` agent then measures
    the performance of Iterative Deepening and the custom heuristic against the
    same opponents.
    
    
    *************************
     Evaluating: ID_Improved 
    *************************
    
    Playing Matches:
    ----------
      Match 1: ID_Improved vs   Random    	Result: 18 to 2
      Match 2: ID_Improved vs   MM_Null   	Result: 14 to 6
      Match 3: ID_Improved vs   MM_Open   	Result: 7 to 13
      Match 4: ID_Improved vs MM_Improved 	Result: 10 to 10
      Match 5: ID_Improved vs   AB_Null   	Result: 12 to 8
      Match 6: ID_Improved vs   AB_Open   	Result: 9 to 11
      Match 7: ID_Improved vs AB_Improved 	Result: 11 to 9
    
    
    Results:
    ----------
    ID_Improved         57.86%
    
    *************************
       Evaluating: Student   
    *************************
    
    Playing Matches:
    ----------
      Match 1:   Student   vs   Random    	Result: 19 to 1
      Match 2:   Student   vs   MM_Null   	Result: 17 to 3
      Match 3:   Student   vs   MM_Open   	Result: 11 to 9
      Match 4:   Student   vs MM_Improved 	Result: 12 to 8
      Match 5:   Student   vs   AB_Null   	Result: 17 to 3
      Match 6:   Student   vs   AB_Open   	Result: 15 to 5
      Match 7:   Student   vs AB_Improved 	Result: 13 to 7
    
    
    Results:
    ----------
    Student             74.29%
    