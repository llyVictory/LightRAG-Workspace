// src/stores/workspace.ts
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { listWorkspaces } from '../api/lightrag';

interface Workspace {
  name: string;
  created_at?: string;
  is_default: boolean;
}

interface WorkspaceState {
  workspaces: Workspace[];
  currentWorkspace: string;
  isLoading: boolean;

  fetchWorkspaces: () => Promise<void>;
  setWorkspace: (name: string) => void;
}

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set) => ({
      workspaces: [],
      currentWorkspace: 'default',
      isLoading: false,

      fetchWorkspaces: async () => {
        set({ isLoading: true });
        try {
          const list = await listWorkspaces();
          set({ workspaces: list });
        } catch (error) {
          console.error("Failed to fetch workspaces", error);
        } finally {
          set({ isLoading: false });
        }
      },

      setWorkspace: (name: string) => {
        set({ currentWorkspace: name });
        // 切换工作区后，建议刷新页面或触发数据重载，防止旧数据残留
        setTimeout(() => window.location.reload(), 100);
      },
    }),
    {
      name: 'lightrag-workspace-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ currentWorkspace: state.currentWorkspace }),
    }
  )
);