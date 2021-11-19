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
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  int file_num = fs->superBlock_ptr->file_num;
  u16 free_block_start = fs->superBlock_ptr->free_block_start;
  // linear search file name

  for (int i = 0; i < file_num; i++)
  {
    // 1. find
    if (strcmp(fs->FCB_arr[i].filename, s) == 0) // equal;
    {
      return file_num;
    }
  }
  // 2. not find
  if (op == G_WRITE) // 2.1 op == G_READ
  {
    // create a file
    strcpy(fs->FCB_arr[file_num].filename, s);
    fs->FCB_arr[file_num] = {.modified_time = gtime, .create_time = gtime, .file_size = 0, .start_block = free_block_start};
    gtime++;
    fs->superBlock_ptr->free_block_start++;
    fs->superBlock_ptr->file_num++;
    return file_num + 1;
  }

  // 2.2 return 0
  return -1;
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
  /* Implement read operation here */
}

__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp)
{
  /* Implement write operation here */
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
  /* Implement LS_D and LS_S operation here */
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  /* Implement rm operation here */
}
