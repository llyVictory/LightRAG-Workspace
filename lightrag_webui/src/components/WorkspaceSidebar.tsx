import React, { useEffect, useState } from 'react';
import { useWorkspaceStore } from '@/stores/workspace';
import { createWorkspace, deleteWorkspace } from '@/api/lightrag';
import { Plus, Trash2, Folder, Check } from 'lucide-react';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/Tooltip';
import { toast } from 'sonner';

export const WorkspaceSidebar = () => {
  const { workspaces, currentWorkspace, setWorkspace, fetchWorkspaces } = useWorkspaceStore();
  const [isCreating, setIsCreating] = useState(false);
  const [newWorkspaceName, setNewWorkspaceName] = useState('');

  useEffect(() => {
    fetchWorkspaces();
  }, []);

  const handleCreate = async () => {
    if (!newWorkspaceName.trim()) return;
    try {
      await createWorkspace(newWorkspaceName);
      setNewWorkspaceName('');
      setIsCreating(false);
      fetchWorkspaces();
    } catch (e) {
      console.error(e);
      alert("Failed to create workspace");
    }
  };

  const handleDelete = async (name: string, e: React.MouseEvent) => {
    e.stopPropagation();

    if (name === 'default') return;

    if (!window.confirm(`Are you sure you want to delete workspace "${name}"?\nThis action is irreversible.`)) {
      return;
    }

    try {
      await deleteWorkspace(name);
      toast.success(`Workspace "${name}" deleted`);

      if (currentWorkspace === name) {
        setWorkspace('default');
      }
      fetchWorkspaces();
    } catch (e: any) {
      console.error(e);
      const errorMsg = e.response?.data?.detail || "Failed to delete workspace";
      toast.error(errorMsg);
    }
  };

  return (
    // [修改 1] w-64 固定宽度，移除 hover:w-64 和 transition 动画
    <div className="w-64 h-screen bg-muted/30 border-r flex flex-col flex-shrink-0 z-50">

      {/* Header */}
      <div className="h-14 flex items-center justify-between px-4 border-b shrink-0">
        {/* [修改 2] 移除 opacity-0，文字常显 */}
        <span className="font-bold truncate">
          Workspaces
        </span>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0"
                onClick={() => setIsCreating(true)}
              >
                <Plus className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Create New Workspace</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Create Form */}
      {isCreating && (
        <div className="p-2 border-b bg-background/50 animate-in fade-in slide-in-from-top-2">
          <Input
            autoFocus
            value={newWorkspaceName}
            onChange={(e) => setNewWorkspaceName(e.target.value)}
            placeholder="Name..."
            className="h-8 text-xs mb-2"
            onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === 'Enter' && handleCreate()}
          />
          <div className="flex gap-2">
            <Button size="sm" className="h-7 text-xs flex-1" onClick={handleCreate}>Add</Button>
            <Button size="sm" variant="secondary" className="h-7 text-xs flex-1" onClick={() => setIsCreating(false)}>Cancel</Button>
          </div>
        </div>
      )}

      {/* List */}
      <div className="flex-1 overflow-y-auto py-2 space-y-1">
        {workspaces.map((ws) => (
          <div
            key={ws.name}
            onClick={() => setWorkspace(ws.name)}
            className={`
              relative flex items-center h-10 px-4 cursor-pointer hover:bg-accent/50 transition-colors group
              ${currentWorkspace === ws.name ? 'bg-accent text-accent-foreground' : 'text-muted-foreground'}
            `}
            title={ws.name}
          >
            {/* Icon */}
            <div className="flex items-center justify-center min-w-[1.5rem] mr-2">
              {currentWorkspace === ws.name ?
                <Check className="h-4 w-4 text-primary" /> :
                <Folder className="h-4 w-4 opacity-50" />
              }
            </div>

            {/* Name */}
            {/* [修改 3] 移除 opacity-0，确保文字始终可见 */}
            <div className="flex-1 min-w-0 flex justify-between items-center overflow-hidden">
              <span className="truncate text-sm">{ws.name}</span>

              {/* Delete Button */}
              {/* [修改 4] 移除 opacity-0，改为 opacity-50，悬停变深。确保始终能看到垃圾桶图标 */}
              {ws.name !== 'default' && (
                <button
                  onClick={(e) => handleDelete(ws.name, e)}
                  className="opacity-50 hover:opacity-100 hover:bg-destructive/10 hover:text-destructive p-1 rounded transition-all ml-2"
                  title="Delete Workspace"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              )}
            </div>

            {/* Active Indicator */}
            {currentWorkspace === ws.name && (
              <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
};