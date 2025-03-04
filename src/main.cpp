#include <SFML/Graphics.hpp>
#include <stack>
#include <iostream>
#include <random>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <set>
#include <cmath>
#include <iostream>
#include <SFML/Window/Clipboard.hpp> // Include for clipboard support
#include <fstream>
#include <string>
#include <ctime>
#include <cstdlib>

enum MazeSource { AUTO_GENERATED, FILE_LOADED };
MazeSource mazeSource = AUTO_GENERATED;

#define CELL_WIDTH 20
int SIZE = 9;       //default size is 9x9

// Add these variables
enum GameState { MENU, GENERATING, RUNNING, COMPLETED, GENERATING_CV, END };
GameState gameState = MENU;

int selectedAlgorithm = 0; // 0 for DFS, 1 for A*, 2 for BFS (DFS is default)

bool startButtonAlgoClicked = false;
bool startButtonCVClicked = false;
bool endButtonClicked = false;
bool algorithmDropdownOpen = false;
bool sizeDropdownOpen = false;
bool submittedImg = false;

class Cell {
public:
    int x, y;
    int pos;
    float size = 30.f;
    float thickness = 2.f;
    bool walls[4] = {true, true, true, true};
    bool visited = false;
    bool isActive = false;

    Cell() {}
    Cell(int _x, int _y) : x(_x), y(_y) {}

    void draw(sf::RenderWindow* window);
};

void Cell::draw(sf::RenderWindow* window) {
    sf::RectangleShape rect;

    if (isActive) {
        rect.setFillColor(sf::Color(247, 23, 53));
        rect.setSize(sf::Vector2f(size, size));
        rect.setPosition(x, y);
        window->draw(rect);
    }
    rect.setFillColor(sf::Color(223, 243, 228));

    if (walls[0]) {
        rect.setSize(sf::Vector2f(size, thickness));
        rect.setPosition(x, y);
        window->draw(rect);
    }
    if (walls[1]) {
        rect.setSize(sf::Vector2f(thickness, size));
        rect.setPosition(x + size, y);
        window->draw(rect);
    }
    if (walls[2]) {
        rect.setSize(sf::Vector2f(size + thickness, thickness));
        rect.setPosition(x, y + size);
        window->draw(rect);
    }
    if (walls[3]) {
        rect.setSize(sf::Vector2f(thickness, size));
        rect.setPosition(x, y);
        window->draw(rect);
    }
}

void resetMaze(Cell *maze,int size){
    for (int i = 0; i < size*size; i++) {
        for (int j = 0; j < 4; j++) {
            maze[i].walls[j] = true;
            maze[i].visited = false;
            maze[i].isActive = false;
        }
    }

}

// Function to load a maze from a file
bool loadMazeFromFile(Cell* maze, int size, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // Reset the maze
    resetMaze(maze, size);

    // The file format will be:
    // Line 1: Size of the maze (must match the selected size)
    // Following lines: One line per cell, with 4 binary digits representing walls [top, right, bottom, left]
    // 1 means wall exists, 0 means no wall

    int fileSize;
    file >> fileSize;

    if (fileSize != size) {
        std::cerr << "Maze size in file (" << fileSize << ") doesn't match selected size (" << size << ")" << std::endl;
        return false;
    }

    for (int i = 0; i < size * size; i++) {
        std::string wallConfig;
        file >> wallConfig;

        if (wallConfig.length() != 4) {
            std::cerr << "Invalid wall configuration for cell " << i << std::endl;
            return false;
        }

        maze[i].walls[0] = (wallConfig[0] == '1'); // top
        maze[i].walls[1] = (wallConfig[1] == '1'); // right
        maze[i].walls[2] = (wallConfig[2] == '1'); // bottom
        maze[i].walls[3] = (wallConfig[3] == '1'); // left
    }

    file.close();
    return true;
}

void removeWallsBetween(Cell *maze,Cell *current,Cell *chosen,int size) {
    // top
    if(current->pos-size == chosen->pos){
        current->walls[0] = false;
        chosen->walls[2] = false;
    // right
    } else if(current->pos+1 == chosen->pos){
        current->walls[1] = false;
        chosen->walls[3] = false;
    // bottom 
    } else if(current->pos+size == chosen->pos){
        current->walls[2] = false;
        chosen->walls[0] = false;
    // left 
    } else if(current->pos-1 == chosen->pos){
        current->walls[3] = false;
        chosen->walls[1] = false;
    }
}

void makeMaze(Cell *maze,int size){
    resetMaze(maze,size);
    std::stack<Cell> stack;
    maze[0].visited = true;
    stack.push(maze[0]);

    while(!stack.empty()){
        Cell current = stack.top();
        stack.pop();
        int pos = current.pos;
        std::vector<int> neighbours;

        if((pos) % (size) != 0 && pos > 0){
            Cell left = maze[pos-1]; 
            if(!left.visited){
                neighbours.push_back(pos-1);
            }
        }
        if((pos+1) % (size) != 0 && pos < size * size){
            Cell right = maze[pos+1]; 
            if(!right.visited){
                neighbours.push_back(pos+1);
            }

        }
        if((pos+size) < size*size){
            Cell bottom = maze[pos+size]; 
            if(!bottom.visited){
                neighbours.push_back(pos+size);
            }
        }

        if((pos-size) > 0){
            Cell top = maze[pos-size]; 
            if(!top.visited){
                neighbours.push_back(pos-size);
            }
        }

        if(neighbours.size() > 0){
            // generate a random array index for selecting a neighbour
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<std::mt19937::result_type> dist6(0,neighbours.size()-1); 
            int randneighbourpos = dist6(rng);

            Cell *chosen = &maze[neighbours[randneighbourpos]];

            stack.push(current);

            removeWallsBetween(maze,&maze[current.pos],chosen,size);

            chosen->visited = true;
            stack.push(*chosen);
        }
    }

    // reset visit status for agent to navigate
    for (int i = 0; i < size * size; i++) {
        maze[i].visited = false;
    }
}

class Agent {
public:
    int *pos;
    int goal;
    std::stack<Cell> visitedStack;      // stack to keep visited cells for backtracking

    Agent(int &startPos, int goalPos) : pos(&startPos), goal(goalPos) {}

    // Choose the next cell based on the reward matrix
    int chooseNextMoveDFS(Cell* maze) {
        std::vector<int> neighbours;

        if((*pos) % (SIZE) != 0 && *pos > 0){
            Cell left = maze[*pos-1];
            if(!left.visited && !left.walls[1]){
                neighbours.push_back(*pos-1);
            }
        }
        if((*pos+1) % (SIZE) != 0 && *pos < SIZE * SIZE){
            Cell right = maze[*pos+1]; 
            if(!right.visited && !right.walls[3]){
                neighbours.push_back(*pos+1);
            }

        }
        if((*pos+SIZE) < SIZE*SIZE){
            Cell bottom = maze[*pos+SIZE]; 
            if(!bottom.visited && !bottom.walls[0]){
                neighbours.push_back(*pos+SIZE);
            }
        }

        if((*pos-SIZE) > 0){
            Cell top = maze[*pos-SIZE]; 
            if(!top.visited && !top.walls[2]){
                neighbours.push_back(*pos-SIZE);
            }
        }
        
        // Select the neighbor with the highest reward
        int bestMove = *pos;

        std::cout << "visites stack size: " << visitedStack.size() << std::endl;

        // backtrack and fallback when dead end
        if(neighbours.size() == 0 && !visitedStack.empty()){
            std::cout << "dead end" << std::endl;
            visitedStack.pop();
            if(!visitedStack.empty()){
                Cell current = visitedStack.top();
                std::cout << "previous on stack: " << current.pos << std::endl;
                maze[*pos].isActive = false;   // Deactivate current cell
                *pos = current.pos;
                maze[*pos].isActive = true;    // Activate new cell
                return current.pos; //fallback
            }
        }

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist6(0,neighbours.size()-1); 
        int randneighbourpos = dist6(rng);
        bestMove = neighbours[randneighbourpos];      

        return bestMove;
    }

    int chooseNextMoveAStar(Cell* maze) {
        struct Node {
            int pos;
            float cost;
            float heuristic;
            bool operator<(const Node& other) const {
                return cost + heuristic > other.cost + other.heuristic;
            }
        };

        std::priority_queue<Node> openSet;
        std::map<int, float> gScore; // Cost from start to the current node
        std::map<int, int> cameFrom; // To reconstruct the path

        openSet.push({*pos, 0.0f, 0.0f});
        gScore[*pos] = 0.0f;

        while (!openSet.empty()) {
            Node current = openSet.top();
            openSet.pop();

            if (current.pos == goal) {
                // Reconstruct path 
                int nextMove = goal;
                while (cameFrom.find(nextMove) != cameFrom.end() && cameFrom[nextMove] != *pos) {
                    nextMove = cameFrom[nextMove];
                }
                return nextMove;
            }

            std::vector<int> neighbors;

            // Add valid neighbors
            if (current.pos % SIZE != 0 && !maze[current.pos].walls[3]) neighbors.push_back(current.pos - 1); // Left
            if ((current.pos + 1) % SIZE != 0 && !maze[current.pos].walls[1]) neighbors.push_back(current.pos + 1); // Right
            if (current.pos + SIZE < SIZE * SIZE && !maze[current.pos].walls[2]) neighbors.push_back(current.pos + SIZE); // Down
            if (current.pos - SIZE >= 0 && !maze[current.pos].walls[0]) neighbors.push_back(current.pos - SIZE); // Up

            for (int neighbor : neighbors) {
                float tentativeGScore = gScore[current.pos] + 1.0f; // Cost to move to the neighbor
                if (gScore.find(neighbor) == gScore.end() || tentativeGScore < gScore[neighbor]) {
                    cameFrom[neighbor] = current.pos;
                    gScore[neighbor] = tentativeGScore;
                    float heuristic = std::abs(neighbor % SIZE - goal % SIZE) + std::abs(neighbor / SIZE - goal / SIZE);
                    openSet.push({neighbor, tentativeGScore, heuristic});
                }
            }
        }

        return *pos; // No path found, stay in place
    }


    int chooseNextMoveBFS(Cell* maze) {
        std::queue<int> queue;
        std::map<int, int> cameFrom; // To reconstruct the path
        std::set<int> visited;

        queue.push(*pos);
        visited.insert(*pos);

        while (!queue.empty()) {
            int current = queue.front();
            queue.pop();

            if (current == goal) {
                // Reconstruct path 
                int nextMove = goal;
                while (cameFrom.find(nextMove) != cameFrom.end() && cameFrom[nextMove] != *pos) {
                    nextMove = cameFrom[nextMove];
                }
                return nextMove;
            }

            std::vector<int> neighbors;

            // Add valid neighbors
            if (current % SIZE != 0 && !maze[current].walls[3]) neighbors.push_back(current - 1); // Left
            if ((current + 1) % SIZE != 0 && !maze[current].walls[1]) neighbors.push_back(current + 1); // Right
            if (current + SIZE < SIZE * SIZE && !maze[current].walls[2]) neighbors.push_back(current + SIZE); // Down
            if (current - SIZE >= 0 && !maze[current].walls[0]) neighbors.push_back(current - SIZE); // Up

            for (int neighbor : neighbors) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    queue.push(neighbor);
                    cameFrom[neighbor] = current;
                }
            }
        }

        return *pos; // No path found, stay in place
    }

    int chooseNextMove(Cell* maze) {
        if (selectedAlgorithm == 0) {
            return chooseNextMoveDFS(maze);
        } else if (selectedAlgorithm == 1) {
            return chooseNextMoveAStar(maze);
        } else if (selectedAlgorithm == 2) {
            return chooseNextMoveBFS(maze);
        }
        return *pos; // Default fallback
    }

    void updatePosition(Cell* maze) {
        int nextMove = chooseNextMove(maze);
        std::cout << "Next Move: " << nextMove << std::endl;
        if (nextMove != *pos) {
            maze[*pos].isActive = false;   // Deactivate current cell
            *pos = nextMove;
            maze[*pos].isActive = true;    // Activate new cell
            Cell* cell = &maze[*pos];
            cell->visited = true;         // mark cell as visited
            visitedStack.push(maze[*pos]);
            Cell top = visitedStack.top();
            int topValue = top.pos;
            std::cout << "visited stack top: " << topValue << std::endl;
        }
    }
};


// Helper function to create dropdown options
void createDropdown(sf::RenderWindow& window, const sf::Font& font, const std::vector<std::string>& options, sf::Vector2f position, int& selectedOption, bool& dropdownOpen) {
    sf::RectangleShape dropdownBox(sf::Vector2f(200, 30));
    dropdownBox.setFillColor(sf::Color(100, 100, 250));
    dropdownBox.setPosition(position);

    if (dropdownOpen) {
        // Draw all options
        for (size_t i = 0; i < options.size(); i++) {
            sf::RectangleShape optionBox(sf::Vector2f(200, 30));
            optionBox.setFillColor(i == selectedOption ? sf::Color(150, 150, 250) : sf::Color(100, 100, 250));
            optionBox.setPosition(position.x, position.y + (i + 1) * 30);

            sf::Text optionText(options[i], font, 20);
            optionText.setPosition(position.x + 10, position.y + (i + 1) * 30 + 5);
            window.draw(optionBox);
            window.draw(optionText);
        }
    } else {
        // Draw selected option
        sf::Text selectedText(options[selectedOption], font, 20);
        selectedText.setPosition(position.x + 10, position.y + 5);
        window.draw(dropdownBox);
        window.draw(selectedText);
    }
}

// Main function
int main(int argc, char* argv[]) {
    int currentPos;
    int goalPos;
    Cell* maze;
    Agent* agent;

    sf::RenderWindow window(sf::VideoMode(800, 600), "Maze Solver");
    window.setFramerateLimit(30);

    sf::RectangleShape currentPosRect;
    currentPosRect.setFillColor(sf::Color(166, 207, 213));
    currentPosRect.setSize(sf::Vector2f(CELL_WIDTH, CELL_WIDTH));

    sf::RectangleShape finishRect;
    finishRect.setFillColor(sf::Color(0, 128, 0));
    finishRect.setSize(sf::Vector2f(CELL_WIDTH, CELL_WIDTH));

    // Set up maze, agent, and timer
    sf::Clock clock;
    bool timerStarted = false;
    
    // Setup for button and menu
    sf::Font font;
    if (!font.loadFromFile("arial.ttf")) {
        std::cout << "Error loading font!" << std::endl;
    }
    
    // Define start algo button 
    sf::RectangleShape startButtonAlgo(sf::Vector2f(200, 50));
    startButtonAlgo.setFillColor(sf::Color(100, 100, 250));
    startButtonAlgo.setPosition((window.getSize().x - startButtonAlgo.getSize().x) / 2 - 250, 500);  // Center the button

    sf::Text startButtonAlgoText("Start Algo", font, 24);
    startButtonAlgoText.setPosition((window.getSize().x - startButtonAlgoText.getGlobalBounds().width) / 2 - 250, 510);  // Center the text

    // Define start CV button
    sf::RectangleShape startButtonCV(sf::Vector2f(200, 50));
    startButtonCV.setFillColor(sf::Color(100, 100, 250));
    startButtonCV.setPosition((window.getSize().x - startButtonCV.getSize().x) / 2, 500);  // Center the button

    sf::Text startButtonCVText("Start CV", font, 24);
    startButtonCVText.setPosition((window.getSize().x - startButtonCVText.getGlobalBounds().width) / 2, 510);  // Center the text

    // Define start CV button
    sf::RectangleShape endButton(sf::Vector2f(200, 50));
    endButton.setFillColor(sf::Color(100, 100, 250));
    endButton.setPosition((window.getSize().x - endButton.getSize().x) / 2 + 250, 500);  // Center the button

    sf::Text endButtonText("End", font, 24);
    endButtonText.setPosition((window.getSize().x - endButtonText.getGlobalBounds().width) / 2 + 250, 510);  // Center the text

    // Define input text box
    sf::RectangleShape inputBox(sf::Vector2f(600, 50));
    inputBox.setFillColor(sf::Color::Black);
    inputBox.setOutlineColor(sf::Color(100, 100, 250));
    inputBox.setOutlineThickness(3);
    inputBox.setPosition((window.getSize().x - inputBox.getSize().x) / 2, 200);

    // input text box label
    sf::Text imageInputLabel("Image File Path:", font, 24);
    imageInputLabel.setFillColor(sf::Color::White);
    imageInputLabel.setPosition((window.getSize().x - inputBox.getSize().x) / 2, 160);  // Position above the input box

    // Text entered
    sf::Text inputText;
    inputText.setFont(font);
    inputText.setCharacterSize(14);
    inputText.setFillColor(sf::Color::White);
    inputText.setPosition(inputBox.getPosition().x + 10, inputBox.getPosition().y + 10);

    // Define start CV button
    sf::RectangleShape imgSubmitButton(sf::Vector2f(200, 50));
    imgSubmitButton.setFillColor(sf::Color(100, 100, 250));
    imgSubmitButton.setPosition((window.getSize().x - imgSubmitButton.getSize().x) / 2 + 150, 330);  // Center the button

    sf::Text imgSubmitButtonText("Submit Image", font, 24);
    imgSubmitButtonText.setPosition((window.getSize().x - imgSubmitButtonText.getGlobalBounds().width) / 2 + 150, 340);  // Center the text

    // Define start CV button
    sf::RectangleShape backButton(sf::Vector2f(200, 50));
    backButton.setFillColor(sf::Color(100, 100, 250));
    backButton.setPosition((window.getSize().x - backButton.getSize().x) / 2 - 150, 330);  // Center the button

    sf::Text backButtonText("Back", font, 24);
    backButtonText.setPosition((window.getSize().x - backButtonText.getGlobalBounds().width) / 2 - 150, 340);  // Center the text
    
    std::string imgPath = "";
    bool isTyping = false;

    // Dropdown options
    std::vector<std::string> algorithms = {"DFS", "A*", "BFS"};
    std::vector<std::string> sizes = {"9x9", "16x16", "21x21"};

    int selectedSize = 0; // Default to 9x9

    // Labels for dropdowns
    sf::Text algorithmLabel("Algorithm:", font, 24);
    algorithmLabel.setFillColor(sf::Color::White);
    algorithmLabel.setPosition(30, 5);  // Position above the Algorithm dropdown

    sf::Text sizeLabel("Maze Size:", font, 24);
    sizeLabel.setFillColor(sf::Color::White);
    sizeLabel.setPosition(250, 5);  // Position above the Maze Size dropdown

    while (window.isOpen()) {
        sf::Event event;
        submittedImg = false;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
            
            sf::Vector2f mousePos(event.mouseButton.x, event.mouseButton.y);

            // Handle start algo click
            if (gameState == MENU && event.type == sf::Event::MouseButtonPressed) {
                
                if (startButtonAlgo.getGlobalBounds().contains(mousePos)) {
                    startButtonAlgoClicked = true;
                    gameState = GENERATING;
                }

                if (startButtonCV.getGlobalBounds().contains(mousePos)) {
                    startButtonCVClicked = true;
                    mazeSource = FILE_LOADED;
                    gameState = GENERATING_CV;
                }

                if (endButton.getGlobalBounds().contains(mousePos)) {
                    endButtonClicked = true;
                    gameState = END;
                }

            }

            // Detect click on input box
            if (event.type == sf::Event::MouseButtonPressed) {
                if (inputBox.getGlobalBounds().contains(mousePos)) {
                    isTyping = true;
                } else {
                    isTyping = false;
                }
            }

            // Handle start end button click
            if (event.type == sf::Event::MouseButtonPressed) {
                if (imgSubmitButton.getGlobalBounds().contains(mousePos)) {
                    submittedImg = true;
                }
            }

            // Handle start end button click
            if (event.type == sf::Event::MouseButtonPressed) {
                if (backButton.getGlobalBounds().contains(mousePos)) {
                    gameState = MENU;
                }
            }

            // Handle text input
            if (isTyping && event.type == sf::Event::TextEntered) {
                if (event.text.unicode == 8 && !imgPath.empty()) {  // Backspace
                    imgPath.pop_back();
                } else if (event.text.unicode == 13) {  // Enter key
                    isTyping = false; 
                    std::cout << "Entered Image Path: " << imgPath << std::endl;
                } else if (event.text.unicode < 128) {  // Only handle ASCII
                    imgPath += static_cast<char>(event.text.unicode);
                }
                inputText.setString(imgPath);
            }

            // Handle copy-paste (Ctrl + V)
            if (isTyping && event.type == sf::Event::KeyPressed) {
                if (event.key.control && event.key.code == sf::Keyboard::V) {  
                    std::string clipboardText = sf::Clipboard::getString();
                    imgPath += clipboardText;  // Append clipboard content
                    inputText.setString(imgPath);
                }
            }

            if (gameState == MENU || gameState == GENERATING_CV) {
                // Handle dropdown clicks
                if (event.type == sf::Event::MouseButtonPressed) {
                    // Toggle algorithm dropdown
                    if (sf::FloatRect(30, 30, 200, 30).contains(mousePos)) {
                        algorithmDropdownOpen = !algorithmDropdownOpen;
                        sizeDropdownOpen = false;
                    }

                    // Toggle size dropdown
                    if (sf::FloatRect(250, 30, 200, 30).contains(mousePos)) {  // Adjusted position for size dropdown
                        sizeDropdownOpen = !sizeDropdownOpen;
                        algorithmDropdownOpen = false;
                    }

                    // Handle algorithm selection
                    if (algorithmDropdownOpen) {
                        for (size_t i = 0; i < algorithms.size(); i++) {
                            if (sf::FloatRect(30, 30 + (i + 1) * 30, 200, 30).contains(mousePos)) {
                                selectedAlgorithm = i;
                                algorithmDropdownOpen = false;
                                std::cout << "Selected Algorithm: " << algorithms[selectedAlgorithm] << std::endl;
                            }
                        }
                    }

                    // Handle size selection
                    if (sizeDropdownOpen) {
                        for (size_t i = 0; i < sizes.size(); i++) {
                            if (sf::FloatRect(250, 30 + (i + 1) * 30, 200, 30).contains(mousePos)) {  // Adjusted position for size dropdown
                                selectedSize = i;
                                sizeDropdownOpen = false;
                                SIZE = (selectedSize == 0) ? 9 : (selectedSize == 1) ? 16 : 21;
                                std::cout << "Selected Size: " << SIZE << std::endl;
                            }
                        }
                    }
                }
            }
        }

        window.clear(sf::Color(13, 2, 33));

        if (gameState == MENU) {
            // Draw labels
            window.draw(algorithmLabel);
            window.draw(sizeLabel);

            // Draw dropdown menus
            createDropdown(window, font, algorithms, sf::Vector2f(30, 30), selectedAlgorithm, algorithmDropdownOpen);  // Algorithm dropdown
            createDropdown(window, font, sizes, sf::Vector2f(250, 30), selectedSize, sizeDropdownOpen);  // Maze size dropdown

            window.draw(startButtonAlgo);
            window.draw(startButtonAlgoText);

            window.draw(startButtonCV);
            window.draw(startButtonCVText);  

            window.draw(endButton);
            window.draw(endButtonText);  
    
        } else if (gameState == GENERATING) {
            currentPos = 0;
            goalPos = SIZE * SIZE - 1;
            maze = new Cell[SIZE * SIZE];
            agent = new Agent(currentPos, goalPos);

            for (int i = 30, k = 0; i < CELL_WIDTH * SIZE + 30; i += CELL_WIDTH) {
                for (int j = 30; j < CELL_WIDTH * SIZE + 30; j += CELL_WIDTH, k++) {
                    maze[k].y = i;
                    maze[k].x = j;
                    maze[k].size = CELL_WIDTH;
                    maze[k].pos = k;
                }
            }
            
            if (mazeSource == AUTO_GENERATED) {
                makeMaze(maze, SIZE);
            } else {
                // run python script to create maze.txt from image
                std::string command = "python image_to_maze.py --image_path " + imgPath;
                system(command.c_str());

                imgPath = "";
                inputText.setString("");
                bool success = loadMazeFromFile(maze, SIZE, "maze.txt");
                if (!success) {
                    // If loading fails, fallback to auto-generation
                    std::cout << "Failed to load maze from file, generating automatically" << std::endl;
                    makeMaze(maze, SIZE);
                }
            }

            gameState = RUNNING;
            clock.restart();
            timerStarted = true;
            currentPos = 0;
            maze[currentPos].isActive = true;
            maze[currentPos].visited = true;
        } else if (gameState == RUNNING) {
            if (currentPos == goalPos) {
                gameState = COMPLETED;
            } else {
                agent->updatePosition(maze);
            }

            // Display the maze and agent
            for (int i = 0; i < SIZE * SIZE; i++) {
                maze[i].draw(&window);
            }

            currentPosRect.setPosition(maze[currentPos].x,maze[currentPos].y);
            window.draw(currentPosRect);
            finishRect.setPosition(maze[SIZE*SIZE-1].x,maze[SIZE*SIZE-1].y);
            window.draw(finishRect);

            // Display timer
            sf::Text timerText;
            timerText.setFont(font);
            timerText.setCharacterSize(24);
            timerText.setFillColor(sf::Color::White);
            float elapsedTime = clock.getElapsedTime().asSeconds();
            timerText.setString("Time: " + std::to_string(elapsedTime) + "s");
            window.draw(timerText);

        } else if (gameState == COMPLETED) {
            timerStarted = false;
            std::cout << "Maze solved in: " << clock.getElapsedTime().asSeconds() << " seconds." << std::endl;
            gameState = MENU; // Restart game state to menu after completion
        } else if(gameState == GENERATING_CV){
            window.draw(inputBox);
            window.draw(inputText);
            window.draw(backButton);
            window.draw(backButtonText);

            window.draw(imageInputLabel);
            window.draw(imgSubmitButton);
            window.draw(imgSubmitButtonText);

            // Draw labels
            window.draw(algorithmLabel);
            // Draw dropdown menus
            createDropdown(window, font, algorithms, sf::Vector2f(30, 30), selectedAlgorithm, algorithmDropdownOpen);  // Algorithm dropdown

            if(submittedImg && imgPath != ""){
                gameState = GENERATING;
            }
        }
        else if(gameState == END){
            window.close();
            return 0;
        }

        window.display();
    }

    delete[] maze;
    return 0;
}

