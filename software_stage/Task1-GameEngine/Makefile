CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -flto -DNDEBUG
CXXFLAGS_DEBUG = -std=c++17 -g -O0 -DDEBUG -fsanitize=address,undefined

ENGINE_DIR = engine
HEADERS = $(wildcard $(ENGINE_DIR)/*.h)

TARGET = chess6x6

all: $(TARGET)

$(TARGET): $(ENGINE_DIR)/main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -I$(ENGINE_DIR) -o $(TARGET) $(ENGINE_DIR)/main.cpp

lib: libchess6x6.so

libchess6x6.so: $(ENGINE_DIR)/engine_api.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -I$(ENGINE_DIR) -shared -fPIC -o libchess6x6.so $(ENGINE_DIR)/engine_api.cpp

selfplay: $(ENGINE_DIR)/selfplay.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -I$(ENGINE_DIR) -o selfplay $(ENGINE_DIR)/selfplay.cpp

datagen: $(ENGINE_DIR)/datagen.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -I$(ENGINE_DIR) -pthread -o datagen $(ENGINE_DIR)/datagen.cpp

datagen_fsf: $(ENGINE_DIR)/datagen_fsf.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -I$(ENGINE_DIR) -pthread -o datagen_fsf $(ENGINE_DIR)/datagen_fsf.cpp

tuner: $(ENGINE_DIR)/tuner.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -I$(ENGINE_DIR) -o tuner $(ENGINE_DIR)/tuner.cpp

debug: $(ENGINE_DIR)/main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS_DEBUG) -I$(ENGINE_DIR) -o $(TARGET)_debug $(ENGINE_DIR)/main.cpp

clean:
	rm -f $(TARGET) $(TARGET)_debug libchess6x6.so selfplay datagen tuner

.PHONY: all clean debug lib
