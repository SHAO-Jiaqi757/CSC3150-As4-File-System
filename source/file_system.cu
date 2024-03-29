﻿#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
// #include <cstring>
// #define debug
// #define sort_debug
// #define case3

__device__ __managed__ u32 gtime = 0;

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
                        int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
                        int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
                        int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
    // init variables
    fs->volume = volume;

    // init constants
    fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
    fs->FCB_SIZE = FCB_SIZE;
    fs->FCB_ENTRIES = FCB_ENTRIES;
    fs->STORAGE_SIZE = VOLUME_SIZE;
    fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
    fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
    fs->MAX_FILE_NUM = MAX_FILE_NUM;
    fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
    fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

    // volume control block: # of blocks, # of free blocks, block size, free block pointers
    init_volume(fs);
}

__device__ void init_volume(FileSystem *fs)
{
    fs->superBlock_ptr = reinterpret_cast<SuperBlock *>(fs->volume);
    fs->superBlock_ptr->free_block_count = fs->MAX_FILE_SIZE / fs->STORAGE_BLOCK_SIZE;
    fs->superBlock_ptr->free_block_start = 0;

    fs->FCB_arr = reinterpret_cast<struct FCB *>(fs->volume + fs->SUPERBLOCK_SIZE);
    fs->fileContent_ptr = reinterpret_cast<uchar *>(fs->volume + fs->FILE_BASE_ADDRESS);
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
    u32 total_file = fs->superBlock_ptr->total_file;
    u16 free_block_start = fs->superBlock_ptr->free_block_start;
    // linear search file name
    int create_time = gtime++;
    for (int i = 0; i < total_file; i++)
    {
        // 1. find
        if (my_strcmp(fs->FCB_arr[i].filename, s) == 0) // equal;
        {
            if (op == G_READ)
                return i;
            else if (op == G_WRITE)
            {
                // rm the file,
                create_time = fs->FCB_arr[i].create_time;
                remove_file(fs, i);
            }
        }
    }
    // 2. create empty file
    if (op == G_WRITE) // 2.1 op == G_READ
    {
        // create a file
        int new_fp = fs->superBlock_ptr->total_file++;
        my_strcpy(fs->FCB_arr[new_fp].filename, s);
#ifdef sort_debug
        printf("COPY fp[%d]: %s-> %s \n", new_fp, s, fs->FCB_arr[new_fp].filename);
#endif
        fs->FCB_arr[new_fp].modified_time = gtime++;
        fs->FCB_arr[new_fp].file_size = 0;
        fs->FCB_arr[new_fp].start_block = fs->superBlock_ptr->free_block_start++;
        fs->FCB_arr[new_fp].create_time = create_time;

        fs->superBlock_ptr->free_block_count--;
#ifdef case3
        printf("create fp: %d, size: %d, start_block: %d \n", fs->superBlock_ptr->total_file - 1, fs->FCB_arr[total_file].file_size, fs->FCB_arr[total_file].start_block);
#endif

        return new_fp;
    }

    return ERROR;
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
    // 1. get physical start address from FCB
    // 2. get the end address from FCB (calculate from file size)
    // 3. read the file from disk to output buffer
    u16 start_block = fs->FCB_arr[fp].start_block;
    u16 file_size = fs->FCB_arr[fp].file_size;
    if (size > file_size)
        size = file_size;
    // u16 end_block = start_block + size / fs->STORAGE_BLOCK_SIZE;
    u32 start_addr = start_block * fs->STORAGE_BLOCK_SIZE;
    u32 end_addr = start_addr + size;
#ifdef debug
    printf("In read >> start_addr = %d, end_addr = %d\n", start_addr, end_addr);
#endif // debug
    for (u32 i = start_addr; i < end_addr; i++)
    {
        output[i - start_addr] = fs->fileContent_ptr[i];
#ifdef debug
        printf("output[%d]: %c \n", i - start_addr, output[i - start_addr]);
#endif
    }
}

__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp)
{

    u16 file_size = fs->FCB_arr[fp].file_size;
    u16 start_block = fs->FCB_arr[fp].start_block;
    u32 start_addr = start_block * fs->STORAGE_BLOCK_SIZE;
    int block_needs = (size + fs->STORAGE_BLOCK_SIZE - 1) / fs->STORAGE_BLOCK_SIZE;
    int origin_blocks = file_size == 0 ? 1 : (file_size + fs->STORAGE_BLOCK_SIZE - 1) / fs->STORAGE_BLOCK_SIZE;

    int free_block_count = fs->superBlock_ptr->free_block_count;

    if (block_needs - origin_blocks > free_block_count) // need extra block exceeding total free blocks
    {
        printf("ERROR: no enough space!\n");
        return ERROR;
    }

    // write to file
#ifdef case3
    printf("start_addr: %d, end_addr: %d \n", start_addr, start_addr + size);
#endif
    for (u32 i = start_addr; i < start_addr + size; i++)
    {
        fs->fileContent_ptr[i] = input[i - start_addr];
    }

    // update FCB_arr
    fs->FCB_arr[fp].file_size = size;
    fs->FCB_arr[fp].modified_time = gtime++;

    // update superBlock_ptr
    int delta_block = block_needs - origin_blocks;
    fs->superBlock_ptr->free_block_count -= delta_block;
    fs->superBlock_ptr->free_block_start += delta_block;

#ifdef sort_debug
    printf("----------\n");
    for (int i = 0; i < fs->superBlock_ptr->total_file; i++)
    {
        printf("[%d].filename: %s \n", i, fs->FCB_arr[i].filename);
    }
#endif
#ifdef case3
    printf("fp: %d, filename: %s, size: %d, free_block: %d \n", fp, fs->FCB_arr[fp].filename, size, fs->superBlock_ptr->free_block_count);
#endif
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
    if (op == LS_S)
    {
        printf("===sort by file size===\n");
        // LS_S list all files name and size in the directory and order by size.
        // If there are several files with the same size, then first create first print.
        sort_file(fs, LS_S);
    }
    else if (op == LS_D)
    {
        printf("===sort by modified time===\n");
        // LS_D list all files name in the directory and order by modified time of files.
        sort_file(fs, LS_D);
    }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
    // linear search filename;
    if (op != RM)
        return;
    u32 fp = fs_open(fs, s, G_READ);
    if (fp == ERROR)
    {
        printf("ERROR: no file exits! \n");
        return;
    }
    remove_file(fs, fp);
}

__device__ void compact(FileSystem *fs, u16 block_num, u16 start_block)
{
    auto dst = fs->fileContent_ptr + start_block * fs->STORAGE_BLOCK_SIZE;
    auto src = fs->fileContent_ptr + (start_block + block_num) * fs->STORAGE_BLOCK_SIZE;
    int count = fs->superBlock_ptr->free_block_start - (start_block + block_num);
    memcpy(dst, src, count * fs->STORAGE_BLOCK_SIZE);
}
__device__ void remove_file(FileSystem *fs, int fp)
{

    u16 start_block = fs->FCB_arr[fp].start_block;
    u16 block_num = fs->FCB_arr[fp].file_size == 0 ? 1 : (fs->FCB_arr[fp].file_size + fs->STORAGE_BLOCK_SIZE - 1) / fs->STORAGE_BLOCK_SIZE;
#ifdef debug
    printf("rm %d, file_size: %d, block_num: %d \n", fp, fs->FCB_arr[fp].file_size, block_num);
#endif
    // update FCB
    for (int i = fp; i < fs->superBlock_ptr->total_file - 1; i++)
    {
        fs->FCB_arr[i] = fs->FCB_arr[i + 1];
        fs->FCB_arr[i].start_block -= block_num;
    }
    // update FILE CONTENT;
    if (fp < fs->superBlock_ptr->total_file - 1)
    {
        // compact
        compact(fs, block_num, start_block);
    }

    fs->superBlock_ptr->total_file--;
    fs->superBlock_ptr->free_block_start -= (block_num);
    fs->superBlock_ptr->free_block_count += (block_num);
}
__device__ char *my_strcpy(char *dst, const char *src) //[1]
{
    char *ret = dst; //[3]
    for (size_t i = 0; i < 20; i++)
    {
        if (*src == '\0')
        {
            *dst = *src;
            break;
        }
        *dst = *src;
        ++dst;
        ++src;
    }
    return ret;
}
__device__ int my_strcmp(char *s1, char *s2)
{
    for (int i = 0; i < 20; i++)
    {
        if (s1[i] != s2[i])
            return -1;
        if (s1[i] == '\0' && s2[i] == '\0')
        {
            return 0;
        }
    }
    return -1;
}
__device__ void swap_fcb_blocks(struct FCB *FCB_arr, u32 a_id, u32 b_id)
{
    FCB tmp_fcb;
    tmp_fcb = FCB_arr[a_id];
    FCB_arr[a_id] = FCB_arr[b_id];
    FCB_arr[b_id] = tmp_fcb;
}
__device__ void sort_file(FileSystem *fs, int op)
{
    int file_count = fs->superBlock_ptr->total_file;
    struct FCB *temp_FCB = fs->FCB_arr;
    if (op == LS_S)
    {
        for (int fp = 0; fp < file_count - 1; fp++)
        {
            for (int j = 0; j < file_count - 1 - fp; j++)
            {
                if (temp_FCB[j].file_size < temp_FCB[j + 1].file_size)
                {
                    swap_fcb_blocks(temp_FCB, j, j + 1);
                }
                else if (temp_FCB[j].file_size == temp_FCB[j + 1].file_size)
                {
                    if (temp_FCB[j].create_time > temp_FCB[j + 1].create_time)
                        swap_fcb_blocks(temp_FCB, j, j + 1);
                }
            }
        }
        // print ...
        for (int i = 0; i < file_count; i++)
        {
            printf("%s %u\n", temp_FCB[i].filename, temp_FCB[i].file_size);
        }
    }
    else if (op == LS_D)
    {
        for (int fp = 0; fp < file_count - 1; fp++)
        {
            for (int j = 0; j < file_count - 1 - fp; j++)
            {
                if (temp_FCB[j].modified_time < temp_FCB[j + 1].modified_time)
                {
                    swap_fcb_blocks(temp_FCB, j, j + 1);
                }
            }
        }
        for (int i = 0; i < file_count; i++)
        {
            printf("%s\n", temp_FCB[i].filename);
        }
    }
}
