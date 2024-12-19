#!/usr/bin/env python
# coding: utf-8

import copy
from collections import OrderedDict

class Decode_categories:
    @staticmethod
    def category_one(str_ins):
        # S type
        opcode_registry = {
            "00000": "beq",
            "00001": "bne",
            "00010": "blt",
            "00011": "sw",
        }

        opcode = str_ins[-7:-2]
        imm_4_0 = str_ins[-12:-7]
        funct3 = str_ins[-15:-12]
        rs1 = str_ins[-20:-15]
        rs2 = str_ins[-25:-20]
        imm_11_5 = str_ins[-32:-25]
        imm = imm_11_5 + imm_4_0

        str_opcode = opcode_registry[opcode]
        str_rs1 = "x" + str(int(rs1, 2))
        str_rs2 = "x" + str(int(rs2, 2))

        if str_opcode == "sw":
            immediate = str(binary_to_num_twos_compliment(imm)) + "({})".format(str_rs2)
            decoded_instruction = [str_opcode, str_rs1, immediate]
        else:
            immediate = "#" + str(binary_to_num_twos_compliment(imm))
            decoded_instruction = [str_opcode, str_rs1, str_rs2, immediate]

        return decoded_instruction

    @staticmethod
    def category_two(str_ins):
        # R type
        opcode_registry = {
            "00000": "add",
            "00001": "sub",
            "00010": "and",
            "00011": "or",
        }
        opcode = str_ins[-7:-2]
        rd = str_ins[-12:-7]
        funct3 = str_ins[-15:-12]
        rs1 = str_ins[-20:-15]
        rs2 = str_ins[-25:-20]
        funct7 = str_ins[-32:-25]

        str_opcode = opcode_registry[opcode]
        str_rs1 = "x" + str(int(rs1, 2))
        str_rs2 = "x" + str(int(rs2, 2))
        str_rd = "x" + str(int(rd, 2))
        decoded_instruction = [str_opcode, str_rd, str_rs1, str_rs2]

        return decoded_instruction

    @staticmethod
    def category_three(str_ins):
        # I type
        opcode_registry = {
            "00000": "addi",
            "00001": "andi",
            "00010": "ori",
            "00011": "sll",
            "00100": "sra",
            "00101": "lw",
        }
        opcode = str_ins[-7:-2]
        rd = str_ins[-12:-7]
        funct3 = str_ins[-15:-12]
        rs1 = str_ins[-20:-15]
        imm = str_ins[-32:-20]

        str_opcode = opcode_registry[opcode]
        str_rd = "x" + str(int(rd, 2))
        str_rs1 = "x" + str(int(rs1, 2))
        imm = str(
            binary_to_num_twos_compliment(imm)
        )  ## immediate values, so twos compliment
        decoded_instruction = [str_opcode, str_rd, str_rs1, imm]

        if str_opcode == "lw":
            immediate = imm + "({})".format(str_rs1)
            decoded_instruction = [str_opcode, str_rd, immediate]
        else:
            immediate = "#" + imm
            decoded_instruction = [str_opcode, str_rd, str_rs1, immediate]

        return decoded_instruction

    @staticmethod
    def category_four(str_ins):

        opcode_registry = {"00000": "jal", "11111": "break"}

        opcode = str_ins[-7:-2]
        str_opcode = opcode_registry[opcode]

        if str_opcode == "break":
            decoded_instruction = ("break", True)
        else:
            rd = str_ins[-12:-7]
            imm = str_ins[-32:-12]
            str_rd = "x" + str(int(rd, 2))
            str_imm = "#" + str(binary_to_num_twos_compliment(imm))
            decoded_instruction = ([str_opcode, str_rd, str_imm], False)

        return decoded_instruction


class Isa:
    @staticmethod
    def beq(instruction, pc):
        rs1 = instruction[1]
        rs2 = instruction[2]
        imm = int(instruction[3][1:])  # avoiding the # like in # 20
        imm = imm << 1
        if dict_reg_previous[rs1] == dict_reg_previous[rs2]:
            pc = pc + imm
        else:
            pc = pc + 4

        return pc

    @staticmethod
    def bne(instruction, pc):

        rs1 = instruction[1]
        rs2 = instruction[2]
        imm = int(instruction[3][1:])  # avoiding the # like in # 20
        imm = imm << 1
        if dict_reg_previous[rs1] != dict_reg_previous[rs2]:
            pc = pc + imm
        else:
            pc = pc + 4

        return pc

    @staticmethod
    def blt(instruction, pc):

        rs1 = instruction[1]
        rs2 = instruction[2]
        imm = int(instruction[3][1:])  # avoiding the # like in # 20
        imm = imm << 1
        if dict_reg_previous[rs1] < dict_reg_previous[rs2]:
            pc = pc + imm
        else:
            pc = pc + 4

        return pc

    @staticmethod
    def sw(instruction):
        # sw x5, 324(x6)
        # stores x5 values at adress 324 + x6
        rs1 = instruction[1]
        store_address = int(instruction[2].split("(")[0])
        rs2 = instruction[2].split("(")[1].strip(")")
        store_address = store_address + dict_reg_previous[rs2]
        # updating the memory
        dict_mem_current[store_address] = dict_reg_previous[rs1]

    @staticmethod
    def add(instruction):

        rd = instruction[1]
        rs1 = instruction[2]
        rs2 = instruction[3]
        dict_reg_current[rd] = dict_reg_previous[rs1] + dict_reg_previous[rs2]

    @staticmethod
    def sub(instruction):
        rd = instruction[1]
        rs1 = instruction[2]
        rs2 = instruction[3]
        dict_reg_current[rd] = dict_reg_previous[rs1] - dict_reg_previous[rs2]

    @staticmethod
    def tand(instruction):

        # the bitwise & operation,automatically converts the numbers to twos complimnet
        # AND operation is performed on unsigned values, so taking the absolute values present in registers
        # also storing unsigned
        rd = instruction[1]
        rs1 = instruction[2]
        rs2 = instruction[3]
        result = int(dict_reg_previous[rs1]) & int(dict_reg_previous[rs2])
        # if you want to take twos compliment
        # result  = binary_to_num_twos_compliment(bin(result)[2:])
        dict_reg_current[rd] = result

    @staticmethod
    def tor(instruction):

        # the bitwise | operation,automatically converts the numbers to twos complimnet
        # AND operation is performed on unsigned values, so taking the absolute values present in register

        rd = instruction[1]
        rs1 = instruction[2]
        rs2 = instruction[3]
        result = int(dict_reg_previous[rs1]) | int(dict_reg_previous[rs2])
        # if you want to take twos compliment
        # result  = binary_to_num_twos_compliment(bin(result)[2:])
        dict_reg_current[rd] = result

    # category three
    @staticmethod
    def addi(instruction):

        rd = instruction[1]
        rs1 = instruction[2]
        imm = int(instruction[3][1:])
        dict_reg_current[rd] = dict_reg_previous[rs1] + imm

    @staticmethod
    def andi(instruction):

        rd = instruction[1]
        rs1 = instruction[2]
        # its signed already, during the diassembly, the numebr is already converted to signed
        imm = int(instruction[3][1:])

        # taking in the register as absolute because its unssigned
        # taking the immediate value as signed
        result = int(dict_reg_previous[rs1]) & int(imm)
        # if you want to take twos compliment
        # result  = binary_to_num_twos_compliment(bin(result)[2:])
        dict_reg_current[rd] = result  # int(result, 2)

    @staticmethod
    def ori(instruction):

        rd = instruction[1]
        rs1 = instruction[2]

        # taking in the register as absolute because its unssigned
        # taking the immediate value as signed
        imm = int(instruction[3][1:])
        # 12 bit signed value
        result = int(dict_reg_previous[rs1]) | int(imm)
        # if you want to take twos compliment
        # result  = binary_to_num_twos_compliment(bin(result)[2:])
        dict_reg_current[rd] = result

    @staticmethod
    def sll(instruction):

        rd = instruction[1]
        rs1 = instruction[2]
        imm = int(instruction[3][1:])

        dict_reg_current[rd] = dict_reg_previous[rs1] << imm

    @staticmethod
    def sra(instruction):

        rd = instruction[1]
        rs1 = instruction[2]
        imm = int(instruction[3][1:])

        dict_reg_current[rd] = dict_reg_previous[rs1] >> imm

    @staticmethod
    def lw(instruction):

        rd = instruction[1]
        store_address = int(instruction[2].split("(")[0])
        rs2 = instruction[2].split("(")[1].strip(")")
        store_address = store_address + dict_reg_previous[rs2]

        dict_reg_current[rd] = dict_mem_previous[store_address]

    @staticmethod
    def jal(instruction, pc):

        # jal x7, #4
        rd = instruction[1]
        imm = instruction[2]
        imm = int(imm[1:]) << 1

        # store pc counter
        dict_reg_current[rd] = pc + 4

        pc = pc + imm

        return pc



def classify_category(str_instruction):

    dict_categories = {"00": "c1", "01": "c2", "10": "c3", "11": "c4"}
    return dict_categories[str_instruction[-2:]]


def binary_to_num_twos_compliment(binary_string):
    "takes a binary string and converts them to signed integer as per twos compliment"

    lst_bits = [int(bit) for bit in list(binary_string)]
    lst_base_2 = [2**i for i in range(len(lst_bits))][::-1]
    lst_base_2_conversion = [a * b for a, b in zip(lst_bits, lst_base_2)]

    # checking for signed numbers
    if int(binary_string[0]) == 1:
        # twos compliment
        lst_base_2_conversion[0] = -lst_base_2_conversion[0]

    return sum(lst_base_2_conversion)


def write_disassembly_to_textfile(decoded_instrcution):

    # string formatting
    for i in range(len(decoded_instrcution)):

        temp_op = decoded_instrcution[i][2]

        if type(temp_op) == list:
            temp_op = temp_op[0] + " " + ", ".join(temp_op[1:]) + "\n"
            decoded_instrcution[i][2] = temp_op
        else:
            decoded_instrcution[i][2] = str(temp_op) + "\n"

    # writing diassembly instructions
    file1 = open("disassembly.txt", "w")
    lst_lines = []
    for instruction in decoded_instrcution:

        line = "\t".join(map(str, instruction))
        lst_lines.append(line)

    file1.writelines(lst_lines)
    file1.close()


def write_cylces(file1, cycle_num, pc, instruction, dict_reg, dict_mem):

    instruction = instruction[0] + " " + ", ".join(instruction[1:])

    line1 = ["Cycle {}:".format(cycle_num), str(pc), instruction]

    line1_1 = ["Registers"]

    # makign sure the dictionary is ordered
    lst_ordered_registers = []
    for i in range(32):
        lst_ordered_registers.append(dict_reg["x" + str(i)])

    line2_1 = lst_ordered_registers[:8]
    line2_1.insert(0, "x00:")

    line2_2 = lst_ordered_registers[8:16]
    line2_2.insert(0, "x08:")

    line2_3 = lst_ordered_registers[16:24]
    line2_3.insert(0, "x16:")

    line2_4 = lst_ordered_registers[24:]
    line2_4.insert(0, "x24:")

    line3 = ["Data"]

    dict_mem_orderd = OrderedDict(sorted(dict_mem.items()))

    line4 = []
    list_memeory_values = list(dict_mem_orderd.values())
    list_memeory_adresses = list(dict_mem_orderd.keys())
    step = 8
    for i in range(0, len(dict_mem_orderd), step):

        memory_line = list_memeory_values[i : i + step]
        line_head = list_memeory_adresses[i]
        memory_line.insert(0, str(line_head) + ":")
        line4.append(memory_line)
    # line4[-1].append('\t')

    line5 = ["--------------------"]

    lines = [line5, line1, line1_1, line2_1, line2_2, line2_3, line2_4, line3]
    lines.extend(line4)

    lines_formated = []
    for l in lines:
        lines_formated.append("\t".join(map(str, l)))

    #     file1 = open("mysimulation.txt","a")#append mode
    for line in lines_formated:
        file1.write("{}\n".format(line.strip()))


# ## disassembly


# In[18]:

# creating registers and filling them with 0s

# ![image.png](attachment:image.png)

# In[19]:


def write_cycle(f):

    reg_vales = list(dict_reg_current.values())
    line2_1 = reg_vales[:8]
    line2_1.insert(0, "x00:")
    line2_2 = reg_vales[8:16]
    line2_2.insert(0, "x08:")
    line2_3 = reg_vales[16:24]
    line2_3.insert(0, "x16:")
    line2_4 = reg_vales[24:]
    line2_4.insert(0, "x24:")

    f.write("--------------------\n")
    f.write("Cycle {}:\n".format(itern))
    f.write("\n")

    def format_ins(inst):

        if len(str(inst)) <= 1:
            return ""
        else:
            ##("format", inst)
            return "[" + inst[0] + " " + ", ".join(inst[1:]) + "]"

    f.write("IF Unit:\n")
    f.write(
        "\tWaiting: {}\n".format(
            format_ins(
                dict_cpu_current["q_fetch"]["waiting"]
                if dict_cpu_current["q_fetch"]["waiting"] is not None
                else ""
            )
        )
    )
    f.write(
        "\tExecuted: {}\n".format(
            format_ins(
                dict_cpu_current["q_fetch"]["Execution"]
                if dict_cpu_current["q_fetch"]["Execution"] is not None
                else ""
            )
        )
    )
    f.write("Pre-Issue Queue:\n")
    pre_queue = dict_cpu_current["q_pre_issue"]

    f.write(
        "\tEntry 0: {}\n".format(
            format_ins(pre_queue[0] if len(pre_queue) >= 1 else "")
        )
    )
    f.write(
        "\tEntry 1: {}\n".format(
            format_ins(pre_queue[1] if len(pre_queue) >= 2 else "")
        )
    )
    f.write(
        "\tEntry 2: {}\n".format(
            format_ins(pre_queue[2] if len(pre_queue) >= 3 else "")
        )
    )
    f.write(
        "\tEntry 3: {}\n".format(
            format_ins(pre_queue[3] if len(pre_queue) >= 4 else "")
        )
    )

    f.write("Pre-ALU1 Queue:\n")

    pre_alu1 = dict_cpu_current["q_pre_alu1"]
    f.write(
        "\tEntry 0: {}\n".format(format_ins(pre_alu1[0] if len(pre_alu1) >= 1 else ""))
    )
    f.write(
        "\tEntry 1: {}\n".format(format_ins(pre_alu1[1] if len(pre_alu1) >= 2 else ""))
    )

    f.write(
        "Pre-MEM Queue: {}\n".format(
            format_ins(
                dict_cpu_current["q_pre_mem"][0]
                if len(dict_cpu_current["q_pre_mem"]) >= 1
                else ""
            )
        )
    )
    f.write(
        "Post-MEM Queue: {}\n".format(
            format_ins(
                dict_cpu_current["q_post_mem"][0]
                if len(dict_cpu_current["q_post_mem"]) >= 1
                else ""
            )
        )
    )
    f.write(
        "Pre-ALU2 Queue: {}\n".format(
            format_ins(
                dict_cpu_current["q_pre_alu2"][0]
                if len(dict_cpu_current["q_pre_alu2"]) >= 1
                else ""
            )
        )
    )
    f.write(
        "Post-ALU2 Queue: {}\n".format(
            format_ins(
                dict_cpu_current["q_post_alu2"][0]
                if len(dict_cpu_current["q_post_alu2"]) >= 1
                else ""
            )
        )
    )

    f.write(
        "Pre-ALU3 Queue: {}\n".format(
            format_ins(
                dict_cpu_current["q_pre_alu3"][0]
                if len(dict_cpu_current["q_pre_alu3"]) >= 1
                else ""
            )
        )
    )
    f.write(
        "Post-ALU3 Queue: {}\n".format(
            format_ins(
                dict_cpu_current["q_post_alu3"][0]
                if len(dict_cpu_current["q_post_alu3"]) >= 1
                else ""
            )
        )
    )

    f.write("\n")

    f.write("Registers\n")
    f.write("\t".join(map(str, line2_1)))
    f.write("\n")
    f.write("\t".join(map(str, line2_2)))
    f.write("\n")
    f.write("\t".join(map(str, line2_3)))
    f.write("\n")
    f.write("\t".join(map(str, line2_4)))
    f.write("\n")

    f.write("Data")
    f.write("\n")
    ##(dict_mem_current.values())

    line4 = []

    dict_mem_orderd = OrderedDict(sorted(dict_mem_current.items()))

    line4 = []
    list_memeory_values = list(dict_mem_orderd.values())
    list_memeory_adresses = list(dict_mem_orderd.keys())
    step = 8
    for i in range(0, len(dict_mem_orderd), step):

        memory_line = list_memeory_values[i : i + step]
        line_head = list_memeory_adresses[i]
        memory_line.insert(0, str(line_head) + ":")
        line4.append(memory_line)

    for line in line4:
        f.write("\t".join(map(str, line)))
        f.write("\n")


# In[20]:

# def fetch_stage(dict_cpu_current):

#     #it cares only the instruction is a branch instruction

#     branch_instructions = ['jal', 'beq', 'bne', 'blt']

# def fetch_stage(dict_cpu_current):

#     #it cares only the instruction is a branch instruction

#     branch_instructions = ['jal', 'beq', 'bne', 'blt']

# def fetch_stage(dict_cpu_current):

#     #it cares only the instruction is a branch instruction

#     branch_instructions = ['jal', 'beq', 'bne', 'blt']


def check_for_break(ins):

    flag = False
    if ins[0] == "break":
        return True

    return flag


def check_struct_hazard(dict_cpu_previous):

    return (
        not (len(dict_cpu_previous["q_pre_alu1"]) < 2),
        not (len(dict_cpu_previous["q_pre_alu2"]) < 1),
        not (len(dict_cpu_previous["q_pre_alu3"]) < 1),
    )


def check_branch_RAW_hazards(current_ins):

    if current_ins[0] == "jal":
        return False

    operands_ci = current_ins[1:]
    # filtering out numbers
    operands_ci = [item for item in operands_ci if "#" not in item]

    # print("operands", operands_ci)

    raw_hazard = False
    for item in [
        "q_pre_issue",
        "q_pre_alu1",
        "q_pre_alu2",
        "q_pre_alu3",
        "q_post_alu2",
        "q_post_alu3",
        "q_pre_mem",
        "q_post_mem",
    ]:
        queue = dict_cpu_previous[item]
        # print(queue)
        for ins in queue:
            if ins[1] in operands_ci:
                raw_hazard = True

    # print("function call hazard status :::", raw_hazard)

    return raw_hazard


def normalize_load_store_ins(lst_ins):

    for ins in lst_ins:

        if ins[0] == "lw" or ins[0] == "sw":
            ins[2] = ins[2].split("(")[1].split(")")[0]
            ins.insert(0, 0)

    return lst_ins


def check_data_hazard_ALU1(current_ins, index):

    # RAW
    # during lw x4, 324(x6), only x6 is checked
    # during stote sw x5, 324(x6), both x5, x6 is checked

    print(current_ins[0])

    data_hazard = False
    if current_ins[0] == "lw":
        print("HIT LW")
        operands_ci = [current_ins[2].split("(")[1].split(")")[0]]
    elif current_ins[0] == "sw":
        print("HIT SW")
        operands_ci = [current_ins[1], current_ins[2].split("(")[1].split(")")[0]]

    print(operands_ci)
    # raw
    raw_hazard = False
    for item in [
        "q_pre_issue",
        "q_pre_alu1",
        "q_pre_alu2",
        "q_pre_alu3",
        "q_post_alu2",
        "q_post_alu3",
        "q_pre_mem",
        "q_post_mem",
    ]:

        queue = dict_cpu_current[item]

        if item == "q_pre_issue":
            queue = queue[:index]

        for ins in queue:
            if ins[1] in operands_ci:
                raw_hazard = True

    # WAW
    waw_hazard = False
    operands_ci = [current_ins[1]]

    for item in [
        "q_pre_issue",
        "q_pre_alu1",
        "q_pre_alu2",
        "q_pre_alu3",
        "q_post_alu2",
        "q_post_alu3",
        "q_pre_mem",
        "q_post_mem",
    ]:
        queue = dict_cpu_current[item]
        if item == "q_pre_issue":
            queue = queue[:index]
        for ins in queue:
            if ins[1] in operands_ci:
                waw_hazard = True

    # WAR with only earlier instructions

    #     war_hazard = False
    #     operands_ci = current_ins[1]
    #     temp = dict_cpu_current['q_pre_issue'][:index]
    #     earlier_ins = normalize_load_store_ins(copy.deepcopy(temp))

    #     for ins in earlier_ins:
    #         if operands_ci in ins[2:]:
    #             war_hazard = True

    if waw_hazard or raw_hazard:  # or war_hazard:
        data_hazard = True

    return data_hazard


def check_data_hazard_ALU2_3(current_ins, index):

    data_hazard = False
    # RAW
    # print(current_ins)
    operands_ci = current_ins[2:].copy()
    operands_ci = [item for item in operands_ci if "#" not in item]
    # print("1st operand ci", operands_ci)
    raw_hazard = False
    for item in [
        "q_pre_issue",
        "q_pre_alu1",
        "q_pre_alu2",
        "q_pre_alu3",
        "q_post_alu2",
        "q_post_alu3",
        "q_pre_mem",
        "q_post_mem",
    ]:

        queue = dict_cpu_current[item]
        if item == "q_pre_issue":
            queue = queue[:index]

        for ins in queue:
            # print("2nd operand ci", current_ins[2:])
            if ins[1] in current_ins[2:]:
                raw_hazard = True

        # WAW
    waw_hazard = False
    operands_ci = [current_ins[1]]

    for item in [
        "q_pre_issue",
        "q_pre_alu1",
        "q_pre_alu2",
        "q_pre_alu3",
        "q_post_alu2",
        "q_post_alu3",
        "q_pre_mem",
        "q_post_mem",
    ]:
        queue = dict_cpu_current[item]
        if item == "q_pre_issue":
            queue = queue[:index]
        for ins in queue:
            if ins[1] in current_ins[2:]:
                waw_hazard = True

    #     war_hazard = False
    #     op_ci = current_ins[1]
    #     temp = dict_cpu_current['q_pre_issue'][:index]
    #     earlier_ins2 = normalize_load_store_ins(copy.deepcopy(temp))
    # #         print("WAR HIT")
    # #         print(earlier_ins)

    #     for ins in earlier_ins2:
    #         if op_ci in ins[2:] and (operands_ci != 'lw' and operands_ci != 'sw'):
    #             war_hazard = True

    if raw_hazard or waw_hazard:  # or war_hazard:
        data_hazard = True

    return data_hazard


# In[23]:

if __name__ == "__main__":

    import copy
    from collections import OrderedDict
    import sys

    file = open(sys.argv[1], "r")
    lst_inst = file.readlines()
    lst_inst = [inst.strip() for inst in lst_inst]

    program_counter = 252
    decoded_instrcution = []
    BREAK_FLAG = False
    for instruction in lst_inst:

        # 32 bit instructions so 4byte increments
        program_counter = program_counter + 4

        category = classify_category(instruction)

        if category == "c1" and BREAK_FLAG == False:

            decoded_instrcution.append(
                [
                    instruction,
                    program_counter,
                    Decode_categories.category_one(instruction),
                ]
            )

        elif category == "c2" and BREAK_FLAG == False:

            decoded_instrcution.append(
                [
                    instruction,
                    program_counter,
                    Decode_categories.category_two(instruction),
                ]
            )

        elif category == "c3" and BREAK_FLAG == False:

            decoded_instrcution.append(
                [
                    instruction,
                    program_counter,
                    Decode_categories.category_three(instruction),
                ]
            )

        elif category == "c4" and BREAK_FLAG == False:

            temp_decoded_instr, BREAK_FLAG = Decode_categories.category_four(
                instruction
            )

            decoded_instrcution.append(
                [instruction, program_counter, temp_decoded_instr]
            )

        elif BREAK_FLAG == True:

            decoded_instrcution.append(
                [
                    instruction,
                    program_counter,
                    binary_to_num_twos_compliment(instruction),
                ]
            )

        else:
            raise ValueError("Error: Unmatched record")

    write_disassembly_to_textfile(copy.deepcopy(decoded_instrcution))
    dict_reg = {}
    for i in range(32):
        dict_reg["x" + str(i)] = 0

    # storing all instructions in a dictionary
    dict_ins = {}
    for ins in decoded_instrcution:
        dict_ins[ins[1]] = ins[2]
        if ins[2] == "break":
            break

    # Memory store
    dict_mem = {}
    for ins in decoded_instrcution[::-1]:
        if ins[2] == "break":
            break
        dict_mem[ins[1]] = int(ins[2])

    dispatcher = {
        "beq": Isa.beq,
        "bne": Isa.bne,
        "blt": Isa.blt,
        "sw": Isa.sw,
        "add": Isa.add,
        "sub": Isa.sub,
        "and": Isa.tand,
        "or": Isa.tor,
        "addi": Isa.addi,
        "andi": Isa.andi,
        "ori": Isa.ori,
        "sll": Isa.sll,
        "sra": Isa.sra,
        "lw": Isa.lw,
        "jal": Isa.jal,
    }

    global dict_cpu_previous
    dict_cpu_previous = {
        "q_fetch": {
            "waiting": None,
            "Execution": None,
        },  # first index waiting, second, executed
        "q_pre_issue": [],
        "q_pre_alu1": [],
        "q_pre_alu2": [],
        "q_pre_alu3": [],
        "q_pre_mem": [],
        "q_post_alu2": [],
        "q_post_alu3": [],
        "q_post_mem": [],
    }

    global dict_cpu_current
    dict_cpu_current = copy.deepcopy(dict_cpu_previous)

    global dict_mem_previous, dict_mem_current, dict_reg_previous, dict_reg_current

    dict_mem_previous = dict_mem.copy()
    dict_mem_current = dict_mem.copy()

    dict_reg_previous = dict_reg.copy()
    dict_reg_current = dict_reg.copy()

    pc = 256
    f = open("simulation.txt", "w")

    branch_ins = ["bne", "blt", "beq", "jal", "break"]
    alu_1_opcode = ["lw", "sw"]
    alu_2_opcode = ["add", "addi", "sub"]
    alu_3_opcode = ["and", "andi", "ori", "sll", "sra", "or"]

    immediate_execution_cleanup = False
    pipeline_branch_stall = False
    pc = 256
    itern = 0

    while True:

        itern = itern + 1

        #     if itern == 40:
        #         break

        print("\n\nCYCLE   CYCLE", itern)
        print("PC COUNTER before", pc)
        print("pipeline_stall :::", pipeline_branch_stall)

        current_ins = dict_ins[pc]
        BREAK_FLAG = False
        #     if check_for_break(current_ins):

        #         #dict_cpu_current['q_fetch']['Execution'] = current_ins
        #         BREAK_FLAG = True
        #         write_cycle(f)
        #         break

        if immediate_execution_cleanup and "".join(imm_cleanup_ins) == "".join(
            dict_cpu_current["q_fetch"]["Execution"]
        ):
            print("IMM CLEANUP HIT")
            print("pc after imm hit", pc)
            dict_cpu_current["q_fetch"]["Execution"] = None
            immediate_execution_cleanup = False

        if (
            pipeline_branch_stall is True
            and dict_cpu_previous["q_fetch"]["Execution"] is not None
        ):

            pc = dispatcher[dict_cpu_previous["q_fetch"]["Execution"][0]](
                current_ins, pc
            )
            pipeline_branch_stall = False
            # print("branch pc", pc)
            dict_cpu_current["q_fetch"]["Execution"] = None

        print("PROGRAM COUNTER after", pc)
        break_main_cycle = False
        # loading pre-issue buffer
        skip_cycle = False

        def get_pre_issue_length():

            buffer_size = len(dict_cpu_current["q_pre_issue"])
            if (
                dict_cpu_current["q_fetch"]["Execution"] is None
                and dict_cpu_current["q_fetch"]["waiting"] is None
            ):
                buffer_size = buffer_size + 1
            return buffer_size

        if pipeline_branch_stall is False and BREAK_FLAG is False:
            # two issues in a cycle
            ins_count = 0
            # check if pre issue queue is full
            while get_pre_issue_length() <= 4 and ins_count <= 1:

                # print("issue hits")

                ins_count = ins_count + 1
                print(pc)
                if dict_ins[pc][0] not in branch_ins and dict_ins[pc] != "break":
                    print(dict_ins[pc][0])
                    dict_cpu_current["q_pre_issue"].append(dict_ins[pc])
                    pc = pc + 4

                elif len(dict_ins[pc]) == 5 and dict_ins[pc] == "break":
                    dict_cpu_current["q_fetch"]["Execution"] = [dict_ins[pc]]
                    #                 write_cycle(f)
                    #                 f.close()
                    #                 break_main_cycle = True
                    BREAK_FLAG = True
                    break

                else:
                    # if its a branch instruction checking for branch hazards
                    if check_branch_RAW_hazards(dict_ins[pc]):
                        dict_cpu_current["q_fetch"]["waiting"] = dict_ins[pc]
                        pipeline_branch_stall = True
                    else:
                        print("branch immediate", dict_ins[pc])
                        print("before branch pc", pc)
                        dict_cpu_current["q_fetch"]["Execution"] = dict_ins[pc]
                        pc = dispatcher[dict_ins[pc][0]](dict_ins[pc], pc)
                        print("branch_immediate pc", pc)

                        imm_cleanup_ins = dict_cpu_current["q_fetch"]["Execution"]
                        immediate_execution_cleanup = True
                        pipeline_branch_stall = False

                    break

        if break_main_cycle:
            break

        if (
            pipeline_branch_stall is True
            and dict_cpu_previous["q_fetch"]["waiting"] is not None
            and BREAK_FLAG is False
        ):
            # print("************************FLAg WAITING")

            if not check_branch_RAW_hazards(dict_cpu_previous["q_fetch"]["waiting"]):

                dict_cpu_current["q_fetch"]["Execution"] = dict_cpu_previous["q_fetch"][
                    "waiting"
                ]
                dict_cpu_current["q_fetch"]["waiting"] = None

        # -------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------
        pre_issue_queue = dict_cpu_previous["q_pre_issue"]
        move_alu1_instruction = True
        move_alu2_instruction = True
        move_alu3_instruction = True

        # for index in range(len(pre_issue_queue)):
        index = 0
        loop_restart_flag = False
        while index < len(pre_issue_queue):

            print("ISSUE HIT")
            print(index, len(pre_issue_queue), pre_issue_queue, move_alu1_instruction)
            if (
                index < len(pre_issue_queue)
                and pre_issue_queue[index][0] in alu_1_opcode
                and move_alu1_instruction
            ):
                # check structural hazards
                print("ALU ONEEEEEEEE HIT")
                pre_alu_1_avil, pre_alu_2_avil, pre_alu_3_avil = check_struct_hazard(
                    dict_cpu_previous
                )

                # check RAW or WAW or WAR hazards with active instructions
                data_hazards = check_data_hazard_ALU1(pre_issue_queue[index], index)

                print(
                    "ALU1 status", data_hazards, pre_alu_1_avil, move_alu1_instruction
                )

                #  For MEM instructions, all the source registers are ready at the end of the last cycle.  -- need to figure out

                # ------yet to figure this

                #  The load instruction must wait until all the previous stores are issued. l l s # LW hazard
                # checking if its a load instruction, if it is,
                # then it should make sure that there are no store instructions before
                lw_hazard = False
                if pre_issue_queue[index][0] == "lw":
                    earlier_instructions = pre_issue_queue[:index]
                    for ins in earlier_instructions:
                        if ins[0] == "sw":
                            lw_hazard = True

                #  The stores must be issued in order.
                # if ther is a store instruction before a store instructions, then its sw hazard
                sw_hazard = False
                if pre_issue_queue[index][0] == "sw":
                    earlier_instructions = pre_issue_queue[:index]
                    for ins in earlier_instructions:
                        if ins[0] == "sw":
                            sw_hazard = True

                hazard_status = pre_alu_1_avil or data_hazards or lw_hazard or sw_hazard

                print("final _hazard stats", not hazard_status)

                # all hazard checks
                # no hazard

                if not hazard_status:
                    print("ALU1_push")
                    popped_ins = pre_issue_queue.pop(index)
                    dict_cpu_current["q_pre_issue"].pop(index)
                    dict_cpu_current["q_pre_alu1"].append(popped_ins)
                    move_alu1_instruction = False
                    loop_restart_flag = True

            # pre_issue_queue = dict_cpu_previous['q_pre_issue']
            if (
                index < len(pre_issue_queue)
                and pre_issue_queue[index][0] in alu_2_opcode
                and move_alu2_instruction
            ):

                print("ALU 2 HITs")

                # check structural hazards
                pre_alu_1_avil, pre_alu_2_avil, pre_alu_3_avil = check_struct_hazard(
                    dict_cpu_previous
                )

                # check RAW or WAW hazards with active instructions
                data_hazard = check_data_hazard_ALU2_3(pre_issue_queue[index], index)

                hazard_status = pre_alu_2_avil or data_hazard

                # print("ALU2 data Hazard",data_hazard )
                # print(index, data_hazard, pre_alu_2_avil, pre_issue_queue[index] )

                # no hazard
                if not hazard_status:
                    popped_ins = pre_issue_queue.pop(index)
                    dict_cpu_current["q_pre_issue"].pop(index)
                    # print("******************", popped_ins)
                    dict_cpu_current["q_pre_alu2"].append(popped_ins)
                    move_alu2_instruction = False
                    loop_restart_flag = True

            # pre_issue_queue = dict_cpu_previous['q_pre_issue']

            if (
                index < len(pre_issue_queue)
                and pre_issue_queue[index][0] in alu_3_opcode
                and move_alu3_instruction
            ):
                print("ALU 3 hits")

                # check structural hazards
                pre_alu_1_avil, pre_alu_2_avil, pre_alu_3_avil = check_struct_hazard(
                    dict_cpu_previous
                )

                # check RAW or WAW hazards with active instructions
                data_hazard = check_data_hazard_ALU2_3(pre_issue_queue[index], index)

                hazard_status = pre_alu_3_avil or data_hazard

                print("ALU 3 data hazard", data_hazard)

                # no hazard
                if not hazard_status:
                    popped_ins = pre_issue_queue.pop(index)
                    dict_cpu_current["q_pre_issue"].pop(index)
                    dict_cpu_current["q_pre_alu3"].append(popped_ins)
                    move_alu3_instruction = False
                    loop_restart_flag = True

            index = index + 1
            if loop_restart_flag:
                index = 0
                loop_restart_flag = False

        # write back 1
        if len(dict_cpu_previous["q_pre_mem"]) == 1:

            if dict_cpu_previous["q_pre_mem"][0][0] == "sw":
                instruct = dict_cpu_previous["q_pre_mem"].pop()
                dict_cpu_current["q_pre_mem"].pop()
                # dict_cpu_current['q_post_mem'].pop()
                # back to_previous, any functional units  cannot obtain the new register values written by WB in the same cycle
                dispatcher[instruct[0]](instruct)

        if len(dict_cpu_previous["q_post_mem"]) == 1:
            instruct = dict_cpu_previous["q_post_mem"].pop()
            dict_cpu_current["q_post_mem"].pop()
            # back to_previous, any functional units  cannot obtain the new register values written by WB in the same cycle
            dispatcher[instruct[0]](instruct)

        if len(dict_cpu_previous["q_pre_mem"]) != 0:

            instruct = dict_cpu_previous["q_pre_mem"].pop()
            dict_cpu_current["q_pre_mem"].pop()
            dict_cpu_current["q_post_mem"] = [instruct]

        # write back 2
        # print("ALU2 write backs", dict_cpu_previous['q_post_alu2'])

        if len(dict_cpu_previous["q_post_alu2"]) != 0:
            instruct = dict_cpu_previous["q_post_alu2"].pop()
            dict_cpu_current["q_post_alu2"].pop()
            # print('******************************write back hits alu2')
            dispatcher[instruct[0]](instruct)

        # write back 3
        if len(dict_cpu_previous["q_post_alu3"]) != 0:
            instruct = dict_cpu_previous["q_post_alu3"].pop()
            dict_cpu_current["q_post_alu3"].pop()
            dispatcher[instruct[0]](instruct)

        # move instructions from pre to post

        if len(dict_cpu_previous["q_pre_alu1"]) >= 1:

            instruct = dict_cpu_previous["q_pre_alu1"].pop(0)
            dict_cpu_current["q_pre_alu1"].pop(0)
            dict_cpu_current["q_pre_mem"] = [instruct]

        if len(dict_cpu_previous["q_pre_alu2"]) == 1:

            instruct = dict_cpu_previous["q_pre_alu2"].pop()
            dict_cpu_current["q_pre_alu2"].pop()
            dict_cpu_current["q_post_alu2"] = [instruct]

        # print("General hit, ", dict_cpu_current)

        if len(dict_cpu_previous["q_pre_alu3"]) == 1:
            print("PRE ALU3 TO POST ALU3 HITS")

            instruct = dict_cpu_previous["q_pre_alu3"].pop()
            dict_cpu_current["q_pre_alu3"].pop()
            dict_cpu_current["q_post_alu3"] = [instruct]

        dict_cpu_previous = copy.deepcopy(dict_cpu_current)
        dict_mem_previous = copy.deepcopy(dict_mem_current)
        dict_reg_previous = copy.deepcopy(dict_reg_current)

        #     print( dict_cpu_current)
        #     print(dict_reg_current)f
        write_cycle(f)
        if BREAK_FLAG:
            break

    f.close()

    # print('ALU1 status',data_hazard, pre_alu_1_avil, move_alu1_instruction)
