import sys
import heapq

HALLWAY_LENGTH = 11
FORBIDDEN_HALLWAY_STOPS = {2, 4, 6, 8}
ROOM_TYPES = ['A', 'B', 'C', 'D']
ROOM_DOORS = {'A': 2, 'B': 4, 'C': 6, 'D': 8}
ENERGY_COST = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}


def parse_input(lines):
    if len(lines) < 3:
        raise ValueError("Недостаточно строк ввода")

    hallway_line = lines[1]
    hallway = tuple(hallway_line[1:1 + HALLWAY_LENGTH])
    if len(hallway) != HALLWAY_LENGTH:
        raise ValueError("Некорректная строка коридора")

    room_rows = []
    for row in lines[2:]:
        if len(row) < 10:
            continue
        room_chars = []
        for index in (3, 5, 7, 9):
            symbol = row[index] if index < len(row) else ' '
            if symbol in '.ABCD':
                room_chars.append(symbol)
            else:
                room_chars.append(None)
        if all(char is not None for char in room_chars):
            room_rows.append([char for char in room_chars])
        if row.strip().startswith("#########"):
            break

    if not room_rows:
        raise ValueError("Не найдены строки комнат")

    room_depth = len(room_rows)
    room_columns = []
    for column_index in range(4):
        column = tuple(room_rows[row_index][column_index] for row_index in range(room_depth))
        room_columns.append(column)

    return hallway, tuple(room_columns)


def is_goal_state(state):
    hallway, rooms = state
    if any(symbol != '.' for symbol in hallway):
        return False
    for index, target_type in enumerate(ROOM_TYPES):
        if any(symbol != target_type for symbol in rooms[index]):
            return False
    return True


def is_path_clear(hallway, start_index, end_index):
    if start_index < end_index:
        path = range(start_index + 1, end_index + 1)
    else:
        path = range(end_index, start_index)
    return all(hallway[i] == '.' for i in path)


def is_room_ready(rooms, room_index, amphipod_type):
    return all(cell in ('.', amphipod_type) for cell in rooms[room_index])


def get_deepest_free_slot(rooms, room_index):
    room_column = rooms[room_index]
    for depth in range(len(room_column) - 1, -1, -1):
        if room_column[depth] == '.':
            return depth
    raise RuntimeError("Комната заполнена")


def get_top_amphipod_to_move(rooms, room_index):
    target_type = ROOM_TYPES[room_index]
    room_column = rooms[room_index]
    if all(symbol in ('.', target_type) for symbol in room_column):
        return -1
    for depth in range(len(room_column)):
        if room_column[depth] != '.':
            return depth
    return -1


def apply_move_from_room_to_hallway(state, room_index, room_depth, hallway_position):
    hallway, rooms = state
    amphipod = rooms[room_index][room_depth]
    room_column = list(rooms[room_index])
    room_column[room_depth] = '.'
    new_rooms = list(list(c) for c in rooms)
    new_rooms[room_index] = tuple(room_column)
    new_hallway = list(hallway)
    new_hallway[hallway_position] = amphipod
    return tuple(new_hallway), tuple(tuple(c) for c in new_rooms)


def apply_move_from_hallway_to_room(state, hallway_position, room_index, room_depth):
    hallway, rooms = state
    amphipod = hallway[hallway_position]
    new_hallway = list(hallway)
    new_hallway[hallway_position] = '.'
    room_column = list(rooms[room_index])
    room_column[room_depth] = amphipod
    new_rooms = list(list(c) for c in rooms)
    new_rooms[room_index] = tuple(room_column)
    return tuple(new_hallway), tuple(tuple(c) for c in new_rooms)


def generate_moves_from_room_to_hallway(state):
    hallway, rooms = state
    for room_index in range(4):
        room_depth = get_top_amphipod_to_move(rooms, room_index)
        if room_depth == -1:
            continue
        amphipod = rooms[room_index][room_depth]
        door_position = ROOM_DOORS[ROOM_TYPES[room_index]]
        if any(rooms[room_index][k] != '.' for k in range(0, room_depth)):
            continue
        for hallway_position in range(door_position - 1, -1, -1):
            if hallway[hallway_position] != '.':
                break
            if hallway_position in FORBIDDEN_HALLWAY_STOPS:
                continue
            steps = room_depth + 1 + abs(hallway_position - door_position)
            cost = steps * ENERGY_COST[amphipod]
            yield apply_move_from_room_to_hallway(state, room_index, room_depth, hallway_position), cost
        for hallway_position in range(door_position + 1, HALLWAY_LENGTH):
            if hallway[hallway_position] != '.':
                break
            if hallway_position in FORBIDDEN_HALLWAY_STOPS:
                continue
            steps = room_depth + 1 + abs(hallway_position - door_position)
            cost = steps * ENERGY_COST[amphipod]
            yield apply_move_from_room_to_hallway(state, room_index, room_depth, hallway_position), cost


def generate_moves_from_hallway_to_room(state):
    hallway, rooms = state
    for hallway_position, amphipod in enumerate(hallway):
        if amphipod == '.':
            continue
        target_room_index = ROOM_TYPES.index(amphipod)
        door_position = ROOM_DOORS[amphipod]
        if not is_path_clear(hallway, hallway_position, door_position):
            continue
        if not is_room_ready(rooms, target_room_index, amphipod):
            continue
        room_depth = get_deepest_free_slot(rooms, target_room_index)
        steps = abs(hallway_position - door_position) + (room_depth + 1)
        cost = steps * ENERGY_COST[amphipod]
        yield apply_move_from_hallway_to_room(state, hallway_position, target_room_index, room_depth), cost


def calculate_heuristic(state):
    hallway, rooms = state
    heuristic_value = 0
    for hallway_position, amphipod in enumerate(hallway):
        if amphipod == '.':
            continue
        door_position = ROOM_DOORS[amphipod]
        steps = abs(hallway_position - door_position) + 1
        heuristic_value += steps * ENERGY_COST[amphipod]
    for room_index, target_type in enumerate(ROOM_TYPES):
        door_position = ROOM_DOORS[target_type]
        room_column = rooms[room_index]
        room_depth = len(room_column)
        for depth in range(room_depth):
            amphipod = room_column[depth]
            if amphipod == '.':
                continue
            if amphipod == target_type and all(room_column[k] == target_type for k in range(depth + 1, room_depth)):
                continue
            target_door = ROOM_DOORS[amphipod]
            steps = (depth + 1) + abs(door_position - target_door) + 1
            heuristic_value += steps * ENERGY_COST[amphipod]
    return heuristic_value


def get_neighbors(state):
    for next_state, move_cost in generate_moves_from_hallway_to_room(state):
        yield next_state, move_cost
    for next_state, move_cost in generate_moves_from_room_to_hallway(state):
        yield next_state, move_cost


def run_astar(start_state):
    start_heuristic = calculate_heuristic(start_state)
    priority_queue = [(start_heuristic, 0, start_state)]
    best_cost = {start_state: 0}
    while priority_queue:
        total_cost, current_cost, current_state = heapq.heappop(priority_queue)
        if is_goal_state(current_state):
            return current_cost
        if current_cost != best_cost.get(current_state, float('inf')):
            continue
        for next_state, move_cost in get_neighbors(current_state):
            new_cost = current_cost + move_cost
            if new_cost < best_cost.get(next_state, float('inf')):
                best_cost[next_state] = new_cost
                estimated_total = new_cost + calculate_heuristic(next_state)
                heapq.heappush(priority_queue, (estimated_total, new_cost, next_state))
    return -1


def solve_problem(input_lines):
    start_state = parse_input(input_lines)
    return run_astar(start_state)


def main():
    input_lines = [line.rstrip('\n') for line in sys.stdin]
    while input_lines and not input_lines[0].strip():
        input_lines.pop(0)
    while input_lines and not input_lines[-1].strip():
        input_lines.pop()
    result = solve_problem(input_lines)
    print(result)


if __name__ == "__main__":
    main()
