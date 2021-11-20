#include "file_system.h"
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

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
  fs->superBlock_ptr->free_block_count = fs->STORAGE_SIZE / fs->STORAGE_BLOCK_SIZE;
  fs->superBlock_ptr->free_block_start = 0;

  fs->FCB_arr = reinterpret_cast<FCB *>(fs->volume + 4 * 1024);

  fs->fileContent_ptr = reinterpret_cast<uchar *>(fs->volume + fs->FILE_BASE_ADDRESS);
#ifdef DEBUG
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
    if (strcmp(fs->FCB_arr[i].filename, s) == 0) // equal;
    {
      if (op == G_READ)
        return i;
      else if (op = G_WRITE)
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
    strcpy(fs->FCB_arr[file_num].filename, s);
    fs->FCB_arr[file_num] = {.modified_time = gtime, .file_size = 0, .start_block = free_block_start};
    fs->FCB_arr[file_num].create_time = create_time;
    gtime++;
    fs->superBlock_ptr->free_block_start++;
    fs->superBlock_ptr->free_block_count--;
    fs->superBlock_ptr->file_num++;
    return fs->superBlock_ptr->file_num - 1;
  }

  // 2.2 return 0
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
  for (u32 i = start_addr; i < end_addr; i++)
  {
    output[i - start_addr] = *(fs->fileContent_ptr + i);
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp)
{
  u16 file_size = fs->FCB_arr[fp].file_size;
  u32 create_time = fs->FCB_arr[fp].create_time;
  char *filename = fs->FCB_arr[fp].filename;
  u16 start_block = fs->FCB_arr[fp].start_block;
  u32 start_addr = start_block * fs->STORAGE_BLOCK_SIZE;
  int block_needs = (size + fs->STORAGE_BLOCK_SIZE - 1) / fs->STORAGE_BLOCK_SIZE;
  int origin_blocks = (file_size + fs->STORAGE_BLOCK_SIZE - 1) / fs->STORAGE_BLOCK_SIZE;

  int free_block_count = fs->superBlock_ptr->free_block_count;
  int free_block_start = fs->superBlock_ptr->free_block_start;

  if (block_needs - origin_blocks > free_block_count) // need extra block exceeding total free blocks
  {
    // not enough space ?
    printf("ERROR: no enough space!\n");
    return ERROR;
  }
  else if (block_needs - origin_blocks != 0) // not match block size
  {
    // can not be expended, need compact
    // remove the file
    fs_gsys(fs, RM, filename);
    // append new file
    // create a file
    int new_fp = fs->superBlock_ptr->file_num;

    strcpy(fs->FCB_arr[new_fp].filename, filename);
    fs->FCB_arr[new_fp] = {.create_time = create_time, .start_block = fs->superBlock_ptr->free_block_start};
    fs->superBlock_ptr->free_block_start += block_needs;
    fs->superBlock_ptr->free_block_count -= block_needs;
    fs->superBlock_ptr->file_num++;

    start_addr = fs->FCB_arr[new_fp].start_block * fs->STORAGE_BLOCK_SIZE;
    fp = new_fp;
  }

  // write to file
  for (u16 i = start_addr; i < start_addr + size; i++)
  {
    *(fs->fileContent_ptr + i) = input[i - start_addr];
  }

  // update FCB_arr
  fs->FCB_arr[fp].file_size = size;
  fs->FCB_arr[fp].modified_time = gtime++;
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
  auto dst = fs->fileContent_ptr + start_block * fs->STORAGE_BLOCK_SIZE;
  auto src = fs->fileContent_ptr + (start_block + block_num) * fs->STORAGE_BLOCK_SIZE;
  int count = fs->superBlock_ptr->free_block_start - (start_block + block_num);
  memcpy(dst, src, count * fs->STORAGE_BLOCK_SIZE);
}
__device__ void remove_file(FileSystem *fs, int fp)
{

  u16 start_block = fs->FCB_arr[fp].start_block;
  u16 block_num = (fs->FCB_arr[fp].file_size + fs->STORAGE_BLOCK_SIZE - 1) / fs->STORAGE_BLOCK_SIZE;

  // update FCB
  for (int i = fp; i < fs->superBlock_ptr->file_num - 1; i++)
  {
    fs->FCB_arr[i] = fs->FCB_arr[i + 1];
    fs->FCB_arr[i].start_block -= block_num;
  }
  // update FILE CONTENT;
  if (fp < fs->superBlock_ptr->file_num - 1)
  {
    // compact
    compact(fs, block_num, start_block);
  }

  fs->superBlock_ptr->file_num--;
  fs->superBlock_ptr->free_block_start -= (block_num);
  fs->superBlock_ptr->free_block_count += (block_num);
}