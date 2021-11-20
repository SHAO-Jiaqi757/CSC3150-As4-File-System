#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
// #include <cstring>
// #define debug
__device__ __managed__ u32 gtime = 0;

#ifdef debug
__device__ void print_fcb(FileSystem *fs, u32 fp)
{
    printf("create fp: %d, size: %d, start_block: %d \n", fp, fs->FCB_arr[fp].file_size, fs->FCB_arr[fp].start_block);
}
#endif
__device__ u32 size_to_block_number(FileSystem *fs, u32 size)
{
    return size == 0 ? 1 : (size + fs->STORAGE_BLOCK_SIZE - 1) / fs->STORAGE_BLOCK_SIZE;
}
__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
                        int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
                        int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
                        int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
    // init variables
    fs->volume = volume;

    // init constants
    fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;       // 4096
    fs->FCB_SIZE = FCB_SIZE;                     // 32
    fs->FCB_ENTRIES = FCB_ENTRIES;               // 1024
    fs->STORAGE_SIZE = VOLUME_SIZE;              // 1085440=4096+32768+1048576
    fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE; // 32
    fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;   // 20
    fs->MAX_FILE_NUM = MAX_FILE_NUM;             // 1024
    fs->MAX_FILE_SIZE = MAX_FILE_SIZE;           // 1048576
    fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;   // 36864=4096+32768

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
#ifdef debug
    printf("%u\n", fs->fileContent_ptr);
#endif
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
    u32 file_num = fs->superBlock_ptr->file_num;
    u16 free_block_start = fs->superBlock_ptr->free_block_start;
    // linear search file name
    int create_time = gtime;
    for (int i = 0; i < file_num; i++)
    {
        // 1. find

        if (my_strcmp(fs->FCB_arr[i].filename, s) == 0) // equal;
        {
            if (op == G_READ)
                return i;

            else if (op == G_WRITE)
            {
                // rm the file
                create_time = fs->FCB_arr[i].create_time;

#ifdef debug
                printf("rm fsb[%d] \n", i);
                printf("Before remove: file_num=%d, free_block_start=%d free_block_count=%d \n", fs->superBlock_ptr->file_num, fs->superBlock_ptr->free_block_start, fs->superBlock_ptr->free_block_count);
#endif
                remove_file(fs, i);

#ifdef debug
                printf("After remove: file_num=%d, free_block_start=%d free_block_count=%d \n", fs->superBlock_ptr->file_num, fs->superBlock_ptr->free_block_start, fs->superBlock_ptr->free_block_count);
#endif
            }
        }
    }
    // 2. create empty file
    if (op == G_WRITE) // 2.1 op == G_READ
    {
        // create a file
        int new_fp = fs->superBlock_ptr->free_block_start;
        my_strcpy(fs->FCB_arr[new_fp].filename, s);
#ifdef debug
        printf("COPY fp[%d]: %s-> %s \n", new_fp, s, fs->FCB_arr[new_fp].filename);
#endif
        fs->FCB_arr[new_fp].modified_time = gtime;
        fs->FCB_arr[new_fp].file_size = 0;
        fs->FCB_arr[new_fp].start_block = fs->superBlock_ptr->free_block_start;
        fs->FCB_arr[new_fp].create_time = create_time;
        gtime++;
        fs->superBlock_ptr->free_block_start++;
        fs->superBlock_ptr->free_block_count--;
        fs->superBlock_ptr->file_num++;
#ifdef debug
        // printf("create fp: %d, size: %d, start_block: %d \n", file_num, fs->FCB_arr[file_num].file_size, fs->FCB_arr[file_num].start_block);
        print_fcb(fs, new_fp);
#endif
        return file_num;
    }

    // 2.2 return ERROR
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
    u32 start_addr = start_block * fs->STORAGE_BLOCK_SIZE;
    u32 end_addr = start_addr + size;
#ifdef debug
    printf("In read >> start_addr = %d, end_addr = %d\n", start_addr, end_addr);
#endif
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
    u16 origin_size = fs->FCB_arr[fp].file_size;
    // u32 create_time = fs->FCB_arr[fp].create_time;
    // char* filename = fs->FCB_arr[fp].filename;
    u16 start_block = fs->FCB_arr[fp].start_block;
    u32 start_addr = start_block * fs->STORAGE_BLOCK_SIZE;
    // int block_needs = (size + fs->STORAGE_BLOCK_SIZE - 1) / fs->STORAGE_BLOCK_SIZE;
    int block_needs = size_to_block_number(fs, size);
    int origin_blocks = size_to_block_number(fs, origin_size);

    int free_block_count = fs->superBlock_ptr->free_block_count;

    if (block_needs - origin_blocks > free_block_count) // need extra block exceeding total free blocks
    {
        // not enough space ?
        printf("ERROR: no enough space!\n");
        return ERROR;
    }

    // write to file
    for (u16 i = start_addr; i < start_addr + size; i++)
    {
        fs->fileContent_ptr[i] = input[i - start_addr];
    }

    // update FCB_arr
    fs->FCB_arr[fp].file_size = size;
    fs->FCB_arr[fp].modified_time = gtime++;

    // update superBlock_ptr
    int delta_blocks = block_needs - origin_blocks;
    fs->superBlock_ptr->free_block_count -= delta_blocks;
    fs->superBlock_ptr->free_block_start += delta_blocks;
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
    if (op == LS_S)
    {
        printf("===sort by file size===\n");
        // LS_S list all files name and size in the directory and order by size.
        // If there are several files with the same size, then first create first print.
    }
    else if (op == LS_D)
    {
        printf("===sort by modified time===\n");
        // LS_D list all files name in the directory and order by modified time of files.
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
    uchar *dst = fs->fileContent_ptr + start_block * fs->STORAGE_BLOCK_SIZE;
    uchar *src = fs->fileContent_ptr + (start_block + block_num) * fs->STORAGE_BLOCK_SIZE;
    int count = fs->superBlock_ptr->free_block_start - (start_block + block_num);
    memcpy(dst, src, count * fs->STORAGE_BLOCK_SIZE);
}

__device__ void remove_file(FileSystem *fs, int fp)
{
    u16 start_block = fs->FCB_arr[fp].start_block;
    u16 block_num = size_to_block_number(fs, fs->FCB_arr[fp].file_size);
#ifdef debug
    printf("rm %d, file_size: %d, block_num: %d \n", fp, fs->FCB_arr[fp].file_size, block_num);
#endif
    // compact FCB
    for (int i = fp; i < fs->superBlock_ptr->file_num - 1; i++)
    {
        fs->FCB_arr[i] = fs->FCB_arr[i + 1];
        fs->FCB_arr[i].start_block -= block_num;
    }
    // compact if this file is not the last one
    if (fp < fs->superBlock_ptr->file_num - 1)
    {
        compact(fs, block_num, start_block);
    }

    fs->superBlock_ptr->file_num--;
    fs->superBlock_ptr->free_block_start -= block_num;
    fs->superBlock_ptr->free_block_count += block_num;
}

__device__ char *my_strcpy(char *dst, const char *src)
{
    char *ret = dst;
    for (size_t i = 0; i < 20; i++)
    {
        if (*src == '\0')
            break;
        *dst = *src;
        ++dst;
        ++src;
    }
    return ret;
}

__device__ int my_strcmp(char *s1, char *s2)
{
    int i = 0;
    while (1)
    {
        if (s1[i] != s2[i])
            return -1;
        if (s1[i] == '\0' && s2[i] == '\0')
        {
            return 0;
        }
        i++;
    }
    return -1;
}
